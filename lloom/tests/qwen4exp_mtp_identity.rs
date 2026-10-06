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
use lloom::models::qwen4exp::Shape as _;
use lloom::models::qwen4exp::{Config, MtpDrafter, Qwen4Exp, Resident, State};
use lloom::nn::model::CausalLM;
use lloom::nn::speculator::{Anchor, Draft, KvSource, NextnHidden, Probes, SeqId, Speculator};
use lloom::nn::{DecodeOptions, IdentityProposer, Session};
use lloom::profile::{CrownExact, RewindableExact};
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
    model: &Qwen4Exp<RewindableExact, Resident>,
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
    model: &Qwen4Exp<RewindableExact, Resident>,
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

fn speculative<S: Speculator<Probe = NextnHidden>>(
    model: &Qwen4Exp<RewindableExact, Resident>,
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
    model: &Qwen4Exp<RewindableExact, Resident>,
    cfg: &Config,
    spec: &mut S,
    prompt: &[u32],
    max: usize,
    ring: Option<usize>,
) -> Result<(Vec<u32>, bool, usize)> {
    let dev = Device::Cpu;
    let mut cache: State<RewindableExact> = match ring {
        Some(k) => State::<RewindableExact>::in_ring(cfg, &model.spec_pools(1, k, &dev))?,
        None => RewindableExact::single(&model, &dev, 8)?,
    };
    cache.want_nextn = true;
    let mut session = Session::new(model, cache);
    let eos = [cfg.eos_token_id];
    let mut rb = Blocking::new(&eos, &dev)?;
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

#[test]
fn mtp_speculative_decode_is_byte_identical_to_greedy() -> Result<()> {
    let cfg = config();
    let dev = Device::Cpu;
    let model = testing::resident_model::<RewindableExact>(&cfg, &dev);
    let max = 24;
    let (prompt, ref_ids, ref_eos) = long_prompt(&model, &cfg, max)?;

    // (a) the real MTP drafter on a synthetic one-layer head
    let head =
        testing::mtp_head::<CrownExact, _>(&cfg, &dev, testing::resident_experts(&cfg, 7, &dev));
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
        let head = testing::mtp_head::<CrownExact, _>(
            &cfg,
            &dev,
            testing::resident_experts(&cfg, 7, &dev),
        );
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
        let head = testing::mtp_head::<CrownExact, _>(
            &cfg,
            &dev,
            testing::resident_experts(&cfg, 7, &dev),
        );
        let mut seq =
            MtpDrafter::new(&head, model.embed_tokens(), model.lm_head(), &cfg, &dev).with_k(k);
        let (ids_s, eos_s, acc_s) = speculative_on(&model, &cfg, &mut seq, &prompt, max, None)?;
        let head = testing::mtp_head::<CrownExact, _>(
            &cfg,
            &dev,
            testing::resident_experts(&cfg, 7, &dev),
        );
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
    let mut model = testing::resident_model::<RewindableExact>(&cfg, &dev);
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
    let block = PleLayer::<RewindableExact>::new(
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
    let head =
        testing::mtp_head::<CrownExact, _>(&cfg, &dev, testing::resident_experts(&cfg, 7, &dev));
    let mut mtp = MtpDrafter::new(&head, model.embed_tokens(), model.lm_head(), &cfg, &dev);
    let (ids, _, _) = speculative(&model, &cfg, &mut mtp, &prompt, max)?;
    assert_eq!(ids, ref_ids, "MTP drafter with PLE: not the greedy stream");
    Ok(())
}

/// The identity with a prompt past the head's QSA dense range (the tiny
/// indexer's 19 positions): the drafter's seeding observe keys the prompt's
/// rows through the head's chunk prefill past the range (#897's path), and
/// every round after it keys past it — the shape the fixture could not see
/// while its head had no indexer.
#[test]
fn mtp_identity_holds_with_a_prompt_past_the_dense_range() -> Result<()> {
    let cfg = config();
    let dev = Device::Cpu;
    let model = testing::resident_model::<RewindableExact>(&cfg, &dev);
    // `Indexer::dense_limit` = budget + ratio − 1 (19 on the tiny template).
    let limit = cfg.indexer_budget + cfg.indexer_compress_ratio - 1;
    let prompt: Vec<u32> = (0..limit + 11).map(|i| (i as u32 * 7 + 1) % 120).collect();
    let max = 24;
    let (ref_ids, ref_eos) = greedy(&model, &cfg, &prompt, max)?;
    assert!(ref_ids.len() >= 6, "a stream of {} tokens", ref_ids.len());
    for k in [1usize, 2] {
        let head = testing::mtp_head::<CrownExact, _>(
            &cfg,
            &dev,
            testing::resident_experts(&cfg, 7, &dev),
        );
        let mut mtp =
            MtpDrafter::new(&head, model.embed_tokens(), model.lm_head(), &cfg, &dev).with_k(k);
        let (ids, eos, accepted) = speculative(&model, &cfg, &mut mtp, &prompt, max)?;
        assert_eq!(
            ids, ref_ids,
            "k = {k}: MTP drafter past the dense range: not the greedy stream"
        );
        assert_eq!(eos, ref_eos);
        assert!(
            mtp.rows_seen() > limit,
            "k = {k}: the head was keyed with {} rows, inside the range of {limit}",
            mtp.rows_seen()
        );
        println!(
            "MTPPASTDENSE k={k} limit={limit} prompt={} accepted={accepted}",
            prompt.len()
        );
    }
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
    let model = testing::resident_model::<RewindableExact>(&cfg, &dev);
    let (prompt, ref_ids, _) = long_prompt(&model, &cfg, 6)?;
    assert!(ref_ids.len() >= 3);
    let anchor_tok = ref_ids[0]; // the prefill's own argmax
    let right = ref_ids[1]; // greedy's next token after the anchor
    let wrong = if right == 0 { 1 } else { 0 };

    let checksums_after = |tokens: &[u32]| -> Result<Vec<(String, u64, usize)>> {
        let mut cache = RewindableExact::single(&model, &dev, 8)?;
        let input = Tensor::from_vec(tokens.to_vec(), (1, tokens.len()), &dev)?;
        model.forward(&input, 0, &mut cache)?;
        cache.checksums()
    };

    for (draft, expect_accepted) in [(wrong, 1usize), (right, 2usize)] {
        let mut cache = RewindableExact::single(&model, &dev, 8)?;
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
            let mut c = RewindableExact::single(&model, &dev, 8)?;
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

/// MTP at B > 1: two sequences with different prompts through the batched
/// speculative loop (`Session::decode_spec_batched`: one drafter holding a
/// lane per sequence, the rows' verifies through `Probes::verify_rows_*`).
/// Each row's stream must be its own greedy stream, at every K, and each
/// sequence's drafter lane keyed with its own prompt and accepted rows — a
/// lane crossed with another's keys the wrong rows while the verify still
/// corrects every draft, so identity alone would not see it.
#[test]
fn mtp_rows_are_each_their_greedy_stream() -> Result<()> {
    let cfg = config();
    let dev = Device::Cpu;
    let model = testing::resident_model::<RewindableExact>(&cfg, &dev);
    let max = 16;
    let (p0, r0, _) = long_prompt(&model, &cfg, max)?;
    let p1: Vec<u32> = p0.iter().map(|t| (t + 13) % 120).collect();
    let (r1, _) = greedy(&model, &cfg, &p1, max)?;
    assert_ne!(r0, r1, "the two prompts must give two streams");
    for k in [1usize, 2] {
        let head = testing::mtp_head::<CrownExact, _>(
            &cfg,
            &dev,
            testing::resident_experts(&cfg, 7, &dev),
        );
        let mut mtp =
            MtpDrafter::with_seqs(&head, model.embed_tokens(), model.lm_head(), &cfg, &dev, 2)
                .with_k(k);
        let mut sessions: Vec<Session<'_, Qwen4Exp<RewindableExact, Resident>>> = (0..2)
            .map(|_| {
                let mut cache = RewindableExact::single(&model, &dev, 8)?;
                cache.want_nextn = true;
                Ok(Session::new(&model, cache))
            })
            .collect::<Result<_>>()?;
        let prompts: Vec<&[u32]> = vec![&p0, &p1];
        let outs = Session::decode_spec_batched(
            &mut sessions,
            lloom::profile::RewindProof::of(),
            &mut mtp,
            &prompts,
            &DecodeOptions::new(max),
            &[cfg.eos_token_id],
            &dev,
            &mut |_, _| Ok(()),
        )?;
        assert_eq!(outs[0].ids, r0, "k = {k}: row 0 is not its greedy stream");
        assert_eq!(outs[1].ids, r1, "k = {k}: row 1 is not its greedy stream");
        for (seq, (p, r)) in [(&p0, &r0), (&p1, &r1)].into_iter().enumerate() {
            assert!(
                mtp.seq_rows_seen(SeqId(seq)) >= p.len() + r.len() - 1,
                "k = {k}: sequence {seq}'s lane keyed with {} rows for {} prompt + {} generated",
                mtp.seq_rows_seen(SeqId(seq)),
                p.len(),
                r.len()
            );
        }
        println!(
            "MTPROWS k={k} rows_seen={:?}",
            (mtp.seq_rows_seen(SeqId(0)), mtp.seq_rows_seen(SeqId(1)))
        );
    }
    Ok(())
}

/// A turn's stream, whether it ended on EOS, and `(drafter context, target
/// position)`.
type Turn = (Vec<u32>, bool, Option<(usize, usize)>);

/// One turn of the speculative body on a kept session: the stream, whether
/// it ended on EOS, and the drafter's context against the target's position.
fn turn(
    session: &mut Session<'_, Qwen4Exp<RewindableExact, Resident>>,
    spec: &mut MtpDrafter<'_, CrownExact, Resident>,
    cfg: &Config,
    prompt: &[u32],
    max: usize,
) -> Result<Turn> {
    let dev = Device::Cpu;
    let eos = [cfg.eos_token_id];
    let mut rb = Blocking::new(&eos, &dev)?;
    let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
    let out = session.decode_speculative(
        lloom::profile::RewindProof::of(),
        spec,
        &mut rb,
        &mut sampler,
        prompt,
        &DecodeOptions::new(max),
        &eos,
        &mut |_| Ok(()),
    )?;
    Ok((out.ids, out.eos, out.stats.drafter_ctx))
}

/// The engine's cache: the verify's states in phase rings, carries kept.
fn engine_cache(
    model: &Qwen4Exp<RewindableExact, Resident>,
    cfg: &Config,
) -> Result<State<RewindableExact>> {
    let mut cache = State::<RewindableExact>::in_ring(cfg, &model.spec_pools(1, 1, &Device::Cpu))?;
    cache.want_nextn = true;
    Ok(cache)
}

/// A prompt whose `max`-token stream runs to the length cap (no EOS): its
/// last id is the anchor the target has not fed, the re-render's next token.
fn capped_prompt(
    model: &Qwen4Exp<RewindableExact, Resident>,
    cfg: &Config,
    max: usize,
) -> Result<Vec<u32>> {
    for seed in 0..16u32 {
        let prompt: Vec<u32> = (0..6).map(|i| (seed * 13 + i * 5 + 2) % 120).collect();
        let (ids, eos) = greedy(model, cfg, &prompt, max)?;
        if !eos && ids.len() == max {
            return Ok(prompt);
        }
    }
    lloom_tensor::bail!("no prompt ran to the length cap")
}

/// The drafter follows the target's prefix reuse across requests, as the
/// engine runs them: each request builds a drafter over the previous one's
/// lanes (`with_lanes` / `into_lanes`), and on a re-rendered turn the target
/// `Extend`s its cache. The lane is kept for the reused prefix — no reset —
/// and ends the turn at the target's position; the stream is the fresh
/// session's greedy one. A lane keyed from another sequence, and one whose
/// last row was keyed with a token the turn does not continue with, reset
/// and say so.
#[test]
fn the_drafter_follows_the_targets_reuse() -> Result<()> {
    use lloom::nn::ReuseAction;
    let cfg = config();
    let dev = Device::Cpu;
    let model = testing::resident_model::<RewindableExact>(&cfg, &dev);
    let head =
        testing::mtp_head::<CrownExact, _>(&cfg, &dev, testing::resident_experts(&cfg, 7, &dev));
    let drafter = || MtpDrafter::new(&head, model.embed_tokens(), model.lm_head(), &cfg, &dev);
    let n = 8;
    let prompt1 = capped_prompt(&model, &cfg, n)?;
    let tail = [3u32, 9, 27, 4, 11];

    // Turn 1, then the re-render: history + the unfed anchor + a new message.
    let mut session = Session::new(&model, engine_cache(&model, &cfg)?);
    let mut d1 = drafter();
    let (ids1, eos1, ctx1) = turn(&mut session, &mut d1, &cfg, &prompt1, n)?;
    assert!(!eos1 && ids1.len() == n, "turn 1 runs to the cap: {ids1:?}");
    assert_eq!(
        ctx1,
        Some((session.pos(), session.pos())),
        "turn 1: the lane is at the target"
    );
    let anchor = *ids1.last().expect("ids");
    let mut full2 = prompt1.clone();
    full2.extend_from_slice(&ids1);
    full2.extend_from_slice(&tail);
    let prepared = session.prepare(&full2)?;
    assert_eq!(
        prepared.action,
        ReuseAction::Extend,
        "{:?}",
        prepared.action
    );
    assert_eq!(
        prepared.suffix[0], anchor,
        "the suffix opens with the unfed anchor"
    );
    let kept = prepared.reused;
    assert!(kept > prompt1.len(), "turn 2 reuses the reply too: {kept}");

    // Turn 2 on a new drafter over turn 1's lanes, as the engine does.
    let mut d2 = drafter().with_lanes(Some(d1.into_lanes()));
    let (ids2, _, ctx2) = turn(&mut session, &mut d2, &cfg, &prepared.suffix, n)?;
    assert_eq!(
        d2.lane_resets(),
        0,
        "the lane was kept for the reused prefix"
    );
    assert_eq!(
        ctx2,
        Some((session.pos(), session.pos())),
        "turn 2: the lane followed the reuse ({kept} kept) and the suffix"
    );
    assert_eq!(
        d2.cache().kv[0].len(),
        session.pos(),
        "the head's KV holds every position"
    );
    let mut fresh = Session::new(&model, engine_cache(&model, &cfg)?);
    let want = turn(&mut fresh, &mut drafter(), &cfg, &full2, n)?.0;
    let (greedy2, _) = greedy(&model, &cfg, &full2, n)?;
    assert_eq!(want, greedy2, "the reference is greedy's");
    assert_eq!(
        ids2, want,
        "the reused speculative turn is the fresh greedy turn"
    );

    // A lane keyed from another sequence of the same length: reset.
    let other: Vec<u32> = prompt1.iter().map(|t| (t + 17) % 120).collect();
    let mut d_other = drafter();
    turn(
        &mut Session::new(&model, engine_cache(&model, &cfg)?),
        &mut d_other,
        &cfg,
        &other,
        n,
    )?;
    let mut s3 = Session::new(&model, engine_cache(&model, &cfg)?);
    turn(&mut s3, &mut drafter(), &cfg, &prompt1, n)?;
    let p3 = s3.prepare(&full2)?;
    assert_eq!(p3.action, ReuseAction::Extend);
    let mut d3 = drafter().with_lanes(Some(d_other.into_lanes()));
    let (ids3, _, ctx3) = turn(&mut s3, &mut d3, &cfg, &p3.suffix, n)?;
    assert_eq!(d3.lane_resets(), 1, "a lane keyed from other tokens resets");
    let (len, pos) = ctx3.expect("a context");
    assert_eq!(pos, s3.pos());
    assert_eq!(
        len,
        pos - p3.reused,
        "the reset lane holds the suffix and the rounds alone"
    );
    assert_eq!(ids3, want, "output identity holds whatever the lane");

    // The turn continues with a token other than the one the lane's last
    // row was keyed with: that row is the new sequence's no longer — reset.
    let mut s4 = Session::new(&model, engine_cache(&model, &cfg)?);
    let mut d4 = drafter();
    turn(&mut s4, &mut d4, &cfg, &prompt1, n)?;
    let mut full4 = prompt1.clone();
    full4.extend_from_slice(&ids1[..n - 1]);
    full4.push((anchor + 1) % 120);
    full4.extend_from_slice(&tail);
    let p4 = s4.prepare(&full4)?;
    assert_eq!(p4.action, ReuseAction::Extend);
    assert_eq!(p4.reused, kept);
    let mut d4b = drafter().with_lanes(Some(d4.into_lanes()));
    turn(&mut s4, &mut d4b, &cfg, &p4.suffix, n)?;
    assert_eq!(
        d4b.lane_resets(),
        1,
        "a lane whose last kept row read another next token resets"
    );
    println!(
        "MTPREUSE kept={kept} pos={} ctx={ctx2:?} other={ctx3:?}",
        session.pos()
    );
    Ok(())
}

/// `begin_at` cuts a lane back to a shorter reused prefix — its KV, its
/// indexer blocks and its position — when the lane was keyed from those
/// tokens, and resets it for a prefix it was not.
#[test]
fn a_lane_is_cut_back_to_a_reused_prefix_it_was_keyed_from() -> Result<()> {
    let cfg = config();
    let dev = Device::Cpu;
    let model = testing::resident_model::<RewindableExact>(&cfg, &dev);
    let head =
        testing::mtp_head::<CrownExact, _>(&cfg, &dev, testing::resident_experts(&cfg, 7, &dev));
    let mut mtp = MtpDrafter::new(&head, model.embed_tokens(), model.lm_head(), &cfg, &dev);
    // Past one indexer block, so the cut drops a compressed block.
    let prompt: Vec<u32> = (0..13).map(|i| (i * 7 + 3) % 120).collect();
    let mut cache = engine_cache(&model, &cfg)?;
    let logits = model.forward(
        &Tensor::new(prompt.as_slice(), &dev)?.unsqueeze(0)?,
        0,
        &mut cache,
    )?;
    let probe = model.take_probe(&mut cache)?;
    let anchor = logits
        .squeeze(0)?
        .argmax(lloom_tensor::D::Minus1)?
        .to_scalar::<u32>()?;
    mtp.begin(SeqId(0), &prompt)?;
    mtp.observe(SeqId(0), &probe, &[anchor])?;
    assert_eq!(mtp.context_len(SeqId(0)), Some(prompt.len()));
    let ratio = cfg.indexer_compress_ratio;
    assert_eq!(mtp.cache().kv[0].n_blocks(), prompt.len() / ratio);

    // Kept: the first m positions and the token after them are the lane's.
    let m = 6;
    mtp.begin_at(SeqId(0), &prompt[m..m + 2], &prompt[..m])?;
    assert_eq!(mtp.context_len(SeqId(0)), Some(m));
    assert_eq!(mtp.cache().kv[0].len(), m);
    assert_eq!(mtp.cache().kv[0].n_blocks(), m / ratio);
    assert_eq!(mtp.lane_resets(), 0);

    // Another prefix of the same length: reset.
    let mut other = prompt[..m].to_vec();
    other[2] ^= 1;
    mtp.begin_at(SeqId(0), &prompt[m..m + 2], &other)?;
    assert_eq!(mtp.context_len(SeqId(0)), Some(0));
    assert_eq!(mtp.lane_resets(), 1);
    Ok(())
}
