//! The turn checkpoint (issue #894): a request that diverges from the cached
//! sequence one token before the previous prompt's end — the chat re-render,
//! which drops the generation prompt's `<think>` and the whole reply — is a
//! rewind to that position, not a fresh prefill, on the crown's `Shuffle`
//! conv as on the rings; and a drafter with its own context follows the
//! target's reuse instead of restarting at zero.
//!
//! CPU, on `sequence_reuse_tests.rs`'s tiny LFM2 (fixed-ramp weights) and
//! `lfm2_dspark_batched_draft.rs`'s tiny drafter (seeded random weights). The
//! reference in every test is a different mechanism than the subject
//! (`CONTRIBUTING.md` §3.1b): a fresh prefill of the kept prefix against the
//! restored state; a fresh session's greedy stream against the reused
//! speculative one.

use lloom::generation::{Blocking, LogitsProcessor, Sampling};
use lloom::models::lfm2::{
    Config, ConvState, KvAppend, LayerType, Model, Shape as Lfm2Shape, State,
};
use lloom::models::lfm2_dspark::{CtxKv, DSparkConfig, DSparkDrafter, DSparkSpeculator};
use lloom::nn::speculator::{SeqId, Speculator};
use lloom::nn::{DecodeOptions, IdentityProposer, ReuseAction, Session, VarBuilder};
use lloom::profile::{CrownExact, RewindableExact};
use lloom_tensor::{DType, Device, Result, Tensor};
use rand::SeedableRng;
use rand_distr::Distribution;
use std::collections::HashMap;

const HIDDEN: usize = 64;
const VOCAB: usize = 48;

fn target_config(conv_state: ConvState, kv_append: KvAppend, taps: Vec<usize>) -> Config {
    Config {
        vocab_size: VOCAB,
        hidden_size: HIDDEN,
        intermediate_size: 128,
        num_hidden_layers: 6,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        norm_eps: 1e-5,
        rope_theta: 10000.0,
        max_position_embeddings: 512,
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
        eos_token_id: Some(7),
        use_flash_attn: false,
        kv_append,
        conv_state,
        // The chat presets carry the left-context carry; a reused suffix is
        // exact only with it (`sequence-reuse-design.md` §2.5).
        conv_prefill_ctx: true,
        gemv_silu: false,
        gemv_residual: false,
        tap_dtype: DType::F32,
        tap_layers: taps,
        ..Default::default()
    }
}

/// Fixed-ramp weights: deterministic, so two sessions over one model agree
/// exactly and a divergence is the paths'.
fn build<P: Lfm2Shape>(cfg: &Config, dev: &Device) -> Result<Model<P>> {
    let mut tensors = HashMap::new();
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
    Model::<P>::new(cfg, vb)
}

fn drafter_cfg() -> DSparkConfig {
    DSparkConfig {
        hidden_size: HIDDEN,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 32,
        intermediate_size: 96,
        vocab_size: VOCAB,
        rms_norm_eps: 1e-5,
        rope_theta: 1e4,
        rope_is_neox_style: false,
        block_size: 5,
        mask_token_id: 47,
        target_layer_ids: vec![0, 2],
        num_target_layers: 6,
        markov_rank: 8,
        enable_confidence_head: false,
        logits_start: 0,
    }
}

fn randn(rng: &mut rand::rngs::StdRng, shape: Vec<usize>, dev: &Device) -> Result<Tensor> {
    let n: usize = shape.iter().product();
    let normal = rand_distr::Normal::new(0.0f32, 0.2).unwrap();
    let vs: Vec<f32> = (0..n).map(|_| normal.sample(rng)).collect();
    Tensor::from_vec(vs, shape.as_slice(), dev)
}

fn tiny_drafter(dev: &Device, seed: u64) -> Result<DSparkDrafter> {
    let cfg = drafter_cfg();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let h = cfg.hidden_size;
    let hd = cfg.head_dim;
    let q = cfg.num_attention_heads * hd;
    let kv = cfg.num_key_value_heads * hd;
    let mut t: HashMap<String, Tensor> = HashMap::new();
    let mut fill = |name: String, shape: Vec<usize>| -> Result<()> {
        t.insert(name, randn(&mut rng, shape, dev)?);
        Ok(())
    };
    fill("fc.weight".into(), vec![h, cfg.target_layer_ids.len() * h])?;
    fill("hidden_norm.weight".into(), vec![h])?;
    fill("norm.weight".into(), vec![h])?;
    for i in 0..cfg.num_hidden_layers {
        let p = format!("layers.{i}");
        fill(format!("{p}.self_attn.q_proj.weight"), vec![q, h])?;
        fill(format!("{p}.self_attn.k_proj.weight"), vec![kv, h])?;
        fill(format!("{p}.self_attn.v_proj.weight"), vec![kv, h])?;
        fill(format!("{p}.self_attn.o_proj.weight"), vec![h, q])?;
        fill(format!("{p}.self_attn.q_norm.weight"), vec![hd])?;
        fill(format!("{p}.self_attn.k_norm.weight"), vec![hd])?;
        fill(
            format!("{p}.mlp.gate_proj.weight"),
            vec![cfg.intermediate_size, h],
        )?;
        fill(
            format!("{p}.mlp.up_proj.weight"),
            vec![cfg.intermediate_size, h],
        )?;
        fill(
            format!("{p}.mlp.down_proj.weight"),
            vec![h, cfg.intermediate_size],
        )?;
        fill(format!("{p}.input_layernorm.weight"), vec![h])?;
        fill(format!("{p}.post_attention_layernorm.weight"), vec![h])?;
    }
    fill(
        "markov_head.markov_w1.weight".into(),
        vec![cfg.vocab_size, cfg.markov_rank],
    )?;
    fill(
        "markov_head.markov_w2.weight".into(),
        vec![cfg.vocab_size, cfg.markov_rank],
    )?;
    let vb = VarBuilder::from_tensors(t, DType::F32, dev);
    DSparkDrafter::new(cfg, 1024, vb)
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
    let a = a.flatten_all()?.to_vec1::<f32>()?;
    let b = b.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(a.len(), b.len(), "shapes differ");
    Ok(a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max))
}

fn greedy_decode<P: Lfm2Shape>(
    model: &Model<P>,
    session: &mut Session<'_, Model<P>>,
    prompt: &[u32],
    n: usize,
) -> Result<Vec<u32>> {
    let eos = vec![7u32];
    let mut rb = Blocking::new(&eos, &Device::Cpu)?;
    let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
    let _ = model;
    let out = session.decode(
        &mut IdentityProposer,
        &mut rb,
        &mut sampler,
        prompt,
        &DecodeOptions::new(n),
        &mut |_: &[u32]| Ok(()),
    )?;
    Ok(out.ids)
}

/// The chat shape: `prompt1`, a reply, then `full2` = `prompt1` without its
/// last token (the generation prompt's `<think>`), a rendered reply, a new
/// user turn. The divergence is at `prompt1.len() - 1`, past every rewind
/// horizon (the reply plus one). Returns `(prompt1, full2)`.
fn two_turns() -> (Vec<u32>, Vec<u32>) {
    let prompt1: Vec<u32> = (0..24u32).map(|i| (i * 5 + 3) % 46).collect();
    let mut full2 = prompt1[..prompt1.len() - 1].to_vec();
    full2.extend_from_slice(&[11, 12, 13, 14, 2, 21, 22, 23, 24, 25, 26]);
    (prompt1, full2)
}

/// On the crown's `Shuffle × InPlace` and on the rewindable ring, the second
/// turn is `Rewound(reply + 1)` to `prompt1.len() - 1`, and the restored cache
/// — KV length and every conv layer's state — is what a fresh prefill of
/// exactly those tokens leaves. `Cat` has no length to move and stays `Fresh`.
#[test]
fn a_chat_re_render_rewinds_to_the_previous_prompts_checkpoint() -> Result<()> {
    let dev = Device::Cpu;
    let (prompt1, full2) = two_turns();
    let keep = prompt1.len() - 1;
    for (conv_state, kv_append, expect_restore) in [
        (ConvState::Shuffle, KvAppend::InPlace, true),
        (ConvState::RotatingRing { k: 4 }, KvAppend::InPlace, true),
        (ConvState::Shuffle, KvAppend::Cat, false),
    ] {
        let cfg = target_config(conv_state, kv_append, Vec::new());
        let key = cfg.exact_key();
        lloom::profile::dispatch!(key, lloom::profile::LFM2, P => re_render_case::<P>(&cfg, &dev, &prompt1, &full2, keep, expect_restore, conv_state, kv_append))?;
    }
    Ok(())
}

/// One cell of the re-render test, at the profile its config names.
#[allow(clippy::too_many_arguments)]
fn re_render_case<P: Lfm2Shape>(
    cfg: &Config,
    dev: &Device,
    prompt1: &[u32],
    full2: &[u32],
    keep: usize,
    expect_restore: bool,
    conv_state: ConvState,
    kv_append: KvAppend,
) -> Result<()> {
    let model = build::<P>(&cfg, &dev)?;
    let mut session = Session::new(&model, State::<P>::new(true, DType::F32, &cfg, &dev)?);
    let reply = greedy_decode(&model, &mut session, &prompt1, 8)?;
    assert!(
        reply.len() >= 4,
        "{conv_state:?}/{kv_append:?}: the first turn produced a reply"
    );
    let before = session.tokens().len();
    assert!(
        before > keep + 1,
        "the divergence is past the reply: {before} > {keep} + 1"
    );
    let prepared = session.prepare(&full2)?;
    if !expect_restore {
        assert_eq!(
            prepared.action,
            ReuseAction::Fresh,
            "{conv_state:?}/{kv_append:?}"
        );
        return Ok(());
    }
    assert_eq!(
        prepared.action,
        ReuseAction::Rewound(before - keep),
        "{conv_state:?}/{kv_append:?}: the re-render is a rewind to the checkpoint, not a fresh prefill"
    );
    assert_eq!(prepared.reused, keep);
    assert_eq!(prepared.suffix, full2[keep..]);
    assert_eq!(session.pos(), keep);
    assert_eq!(
        session.tokens(),
        &full2[..keep],
        "the token log was cut to the kept prefix"
    );

    // The reference: a fresh prefill of exactly the kept tokens.
    let mut fresh = State::<P>::new(true, DType::F32, &cfg, &dev)?;
    let input = Tensor::new(&full2[..keep], &dev)?.unsqueeze(0)?;
    model.forward(&input, 0, &mut fresh)?;
    assert_eq!(
        session.sequence().cache().kv_len(),
        fresh.kv_len(),
        "KV length"
    );
    assert_eq!(
        session.sequence().cache().conv_phase(),
        fresh.conv_phase(),
        "conv phase"
    );
    for l in 0..cfg.num_hidden_layers {
        let (Some(a), Some(b)) = (
            session.sequence().cache().conv_state_for_test(l),
            fresh.conv_state_for_test(l),
        ) else {
            return Ok(());
        };
        let d = max_abs_diff(&a, &b)?;
        assert!(
            d == 0.0,
            "{conv_state:?}: layer {l} conv state after restore differs from a fresh prefill by {d}"
        );
    }
    // And the continuation is the fresh session's, token for token.
    let mut fresh_session = Session::new(&model, State::<P>::new(true, DType::F32, &cfg, &dev)?);
    let want = greedy_decode(&model, &mut fresh_session, &full2, 12)?;
    let got = greedy_decode(&model, &mut session, &prepared.suffix, 12)?;
    assert_eq!(
        got, want,
        "{conv_state:?}: the reused turn's stream is the fresh turn's"
    );
    Ok(())
}

/// A divergence further back than the checkpoint's tail reaches is still
/// `Fresh`: the checkpoint stands at the pass's end and a bounded number of
/// positions before it, not anywhere.
#[test]
fn a_divergence_past_the_checkpoints_reach_is_a_fresh_prefill() -> Result<()> {
    let dev = Device::Cpu;
    let cfg = target_config(ConvState::Shuffle, KvAppend::InPlace, Vec::new());
    let model = build::<CrownExact>(&cfg, &dev)?;
    let prompt1: Vec<u32> = (0..40u32).map(|i| (i * 5 + 3) % 46).collect();
    let mut session = Session::new(
        &model,
        State::<CrownExact>::new(true, DType::F32, &cfg, &dev)?,
    );
    greedy_decode(&model, &mut session, &prompt1, 4)?;
    let slack = lloom::models::lfm2::TURN_CHECKPOINT_SLACK;
    // Within reach: `slack` before the end.
    let mut near = prompt1[..prompt1.len() - slack].to_vec();
    near.extend_from_slice(&[1, 2, 3]);
    assert!(
        matches!(session.prepare(&near)?.action, ReuseAction::Rewound(_)),
        "{slack} back is in reach"
    );
    // Re-prefill the first turn, then diverge one further back than the reach.
    session.reset();
    greedy_decode(&model, &mut session, &prompt1, 4)?;
    let mut far = prompt1[..prompt1.len() - slack - 1].to_vec();
    far.extend_from_slice(&[1, 2, 3]);
    assert_eq!(
        session.prepare(&far)?.action,
        ReuseAction::Fresh,
        "{} back is past the tail",
        slack + 1
    );
    Ok(())
}

/// The drafter follows the target's reuse: after the re-render's `Rewound`,
/// its context is cut back to the kept prefix and the suffix is keyed after
/// it, so its position equals the target's at the end of the turn with no
/// reset; a slot that never saw the sequence is reset and keys over a gap,
/// counted. The reused speculative stream is the fresh session's greedy one.
#[test]
fn the_drafter_follows_the_targets_reuse() -> Result<()> {
    let dev = Device::Cpu;
    let dcfg = drafter_cfg();
    let k = dcfg.max_draft();
    let cfg = target_config(
        ConvState::RotatingRing { k: k + 1 },
        KvAppend::InPlace,
        dcfg.tap_layers(),
    );
    let model = build::<RewindableExact>(&cfg, &dev)?;
    let (prompt1, full2) = two_turns();
    let keep = prompt1.len() - 1;
    let eos = vec![7u32];
    let n = 12usize;
    let spec_decode = |session: &mut Session<'_, Model<RewindableExact>>,
                       spec: &mut DSparkSpeculator,
                       prompt: &[u32]|
     -> Result<(Vec<u32>, Option<(usize, usize)>)> {
        let mut rb = Blocking::new(&eos, &dev)?;
        let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
        let out = session.decode_speculative(
            lloom::profile::RewindProof::of(),
            spec,
            &mut rb,
            &mut sampler,
            prompt,
            &DecodeOptions::new(n),
            &eos,
            &mut |_: &[u32]| Ok(()),
        )?;
        Ok((out.ids, out.stats.drafter_ctx))
    };

    let mut spec = DSparkSpeculator::new_with_ctx(
        tiny_drafter(&dev, 11)?,
        model.draft_embedding().clone(),
        model.lm_head_linear().clone(),
        1,
        CtxKv::Contiguous,
    )?;
    let mut session = Session::new(
        &model,
        State::<RewindableExact>::new(true, DType::F32, &cfg, &dev)?,
    );
    let out1 = spec_decode(&mut session, &mut spec, &prompt1)?;
    assert_eq!(
        out1.1,
        Some((session.pos(), session.pos())),
        "turn 1: the drafter is at the target's position"
    );
    let prepared = session.prepare(&full2)?;
    assert!(
        matches!(prepared.action, ReuseAction::Rewound(_)),
        "{:?}",
        prepared.action
    );
    assert_eq!(prepared.reused, keep);
    let out2 = spec_decode(&mut session, &mut spec, &prepared.suffix)?;
    assert_eq!(
        spec.ctx_resets(),
        0,
        "the drafter kept its context for the reused prefix"
    );
    assert_eq!(
        out2.1,
        Some((session.pos(), session.pos())),
        "turn 2: the drafter followed the rewind and the suffix"
    );
    assert_eq!(
        spec.ctx_state(SeqId(0)),
        Some((session.pos(), session.pos())),
        "no gap"
    );

    // The reference: a fresh session's greedy stream for `full2`.
    let mut fresh = Session::new(
        &model,
        State::<RewindableExact>::new(true, DType::F32, &cfg, &dev)?,
    );
    let want = greedy_decode(&model, &mut fresh, &full2, n)?;
    assert_eq!(
        out2.0, want,
        "the reused speculative turn is the fresh greedy turn"
    );
    assert!(want.len() > 2, "non-vacuous: {} tokens", want.len());

    // A slot that never saw the sequence: the target reuses, the drafter
    // cannot follow — reset, keyed over a gap, counted, positions still the
    // target's.
    let mut other = DSparkSpeculator::new_with_ctx(
        tiny_drafter(&dev, 11)?,
        model.draft_embedding().clone(),
        model.lm_head_linear().clone(),
        1,
        CtxKv::Contiguous,
    )?;
    let mut session2 = Session::new(
        &model,
        State::<RewindableExact>::new(true, DType::F32, &cfg, &dev)?,
    );
    spec_decode(&mut session2, &mut spec, &prompt1)?;
    let prepared = session2.prepare(&full2)?;
    assert!(matches!(prepared.action, ReuseAction::Rewound(_)));
    let out = spec_decode(&mut session2, &mut other, &prepared.suffix)?;
    assert_eq!(
        other.ctx_resets(),
        1,
        "a slot with another sequence's context resets"
    );
    let (len, offset) = other.ctx_state(SeqId(0)).unwrap();
    assert_eq!(offset, session2.pos(), "the positions are the target's");
    assert!(
        len < offset,
        "the context holds the suffix over a gap: {len} < {offset}"
    );
    assert_eq!(
        out.0, want,
        "output identity holds whatever the drafter's context"
    );
    Ok(())
}

/// `begin_at` keeps a context only for the tokens it was keyed from: the same
/// slot given a different sequence's prefix of the same length resets.
#[test]
fn a_context_keyed_from_other_tokens_is_not_kept() -> Result<()> {
    let dev = Device::Cpu;
    let dcfg = drafter_cfg();
    let cfg = target_config(
        ConvState::RotatingRing {
            k: dcfg.max_draft() + 1,
        },
        KvAppend::InPlace,
        dcfg.tap_layers(),
    );
    let model = build::<RewindableExact>(&cfg, &dev)?;
    let mut spec = DSparkSpeculator::new_with_ctx(
        tiny_drafter(&dev, 11)?,
        model.draft_embedding().clone(),
        model.lm_head_linear().clone(),
        1,
        CtxKv::Contiguous,
    )?;
    let prompt: Vec<u32> = vec![3, 11, 20, 5, 9, 14, 2, 8];
    let mut cache = State::<RewindableExact>::new(true, DType::F32, &cfg, &dev)?;
    let logits = model.forward(
        &Tensor::new(prompt.as_slice(), &dev)?.unsqueeze(0)?,
        0,
        &mut cache,
    )?;
    let taps = cache.take_taps();
    let anchor = logits
        .squeeze(0)?
        .argmax(lloom_tensor::D::Minus1)?
        .to_scalar::<u32>()?;
    spec.begin(SeqId(0), &prompt)?;
    spec.observe(SeqId(0), &taps, &[anchor])?;
    assert_eq!(spec.ctx_state(SeqId(0)), Some((prompt.len(), prompt.len())));
    // The same tokens: kept, cut to the prefix.
    spec.begin_at(SeqId(0), &[40, 41], &prompt[..6])?;
    assert_eq!(spec.ctx_state(SeqId(0)), Some((6, 6)));
    assert_eq!(spec.ctx_resets(), 0);
    // Different tokens of the same length: reset, keyed over a gap.
    let mut other = prompt[..6].to_vec();
    other[2] ^= 1;
    spec.begin_at(SeqId(0), &[40, 41], &other)?;
    assert_eq!(spec.ctx_state(SeqId(0)), Some((0, 6)));
    assert_eq!(spec.ctx_resets(), 1);
    Ok(())
}

/// The steady-state speculative round allocates nothing (issue #894 item 2b):
/// on the drafter's paged context every round after warm-up makes zero pool
/// allocations and zero residency adds, over two turns through `prepare`. The
/// instrument is checked by a control that allocates one fresh-sized buffer
/// per draft and must be seen doing it.
#[cfg(feature = "metal")]
#[test]
fn a_steady_state_round_allocates_nothing_on_the_paged_context() -> Result<()> {
    use lloom::nn::speculator::{Draft, LayerTaps, TargetProbe};

    /// The drafter, with the GPU drained and its completion handlers landed
    /// before every draft: the pool recycles a buffer only once its last
    /// reader has COMPLETED (`DESIGN.md` §6.3b), and the tiny model's
    /// millisecond rounds otherwise race that window and allocate a twin — a
    /// timing property of the pool, not a shape this gate is about. The real
    /// model's 40 ms rounds leave the window closed (the churn line on
    /// `lloom chat -v` reads it). `allocate` makes the control: one buffer of
    /// a new size per draft, so a counter reading zero must read nonzero.
    struct Allocating {
        inner: DSparkSpeculator,
        dev: Device,
        allocate: bool,
        n: usize,
        kept: Vec<Tensor>,
    }
    impl Speculator for Allocating {
        type Probe = LayerTaps<5>;
        type Family = lloom::nn::speculator::DSpark;
        fn n_seq(&self) -> usize {
            self.inner.n_seq()
        }
        fn max_draft(&self) -> usize {
            self.inner.max_draft()
        }
        fn begin(&mut self, seq: SeqId, prompt: &[u32]) -> Result<()> {
            self.inner.begin(seq, prompt)
        }
        fn begin_at(&mut self, seq: SeqId, prompt: &[u32], kept: &[u32]) -> Result<()> {
            self.inner.begin_at(seq, prompt, kept)
        }
        fn draft(&mut self, active: &[SeqId]) -> Result<Vec<Draft>> {
            // Every command buffer complete AND its completion handler run
            // (the clock the pool recycles on is advanced by the handler,
            // on Metal's thread, after `synchronize` returns).
            self.dev.synchronize()?;
            if let Device::Metal(m) = &self.dev {
                let clock = m.gpu_clock();
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
                while clock.completed_count().0 < clock.current_epoch().0 {
                    assert!(
                        std::time::Instant::now() < deadline,
                        "the GPU clock never completed"
                    );
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
            }
            if self.allocate {
                self.n += 1;
                // Held, so the pool cannot hand the size back; a distinct
                // bucket every time.
                self.kept
                    .push(Tensor::zeros((65536 * self.n,), DType::F32, &self.dev)?);
            }
            self.inner.draft(active)
        }
        fn observe(
            &mut self,
            seq: SeqId,
            probe: &<Self::Probe as TargetProbe>::Output,
            accepted: &[u32],
        ) -> Result<()> {
            self.inner.observe(seq, probe, accepted)
        }
        fn context_len(&self, seq: SeqId) -> Option<usize> {
            self.inner.context_len(seq)
        }
    }

    let dev = Device::new_metal(0)?;
    // No eviction during the run: the pool trims free buffers past its byte
    // budget oldest-first (§6.3f), and a size trimmed between two rounds is a
    // fresh allocation at the next round that is the pool's policy, not a
    // size the round lacked. The gate reads the round.
    if let Device::Metal(m) = &dev {
        m.set_free_budget(usize::MAX / 4);
    }
    let dcfg = drafter_cfg();
    let k = dcfg.max_draft();
    let mut cfg = target_config(
        ConvState::RotatingRing { k: k + 1 },
        KvAppend::InPlace,
        dcfg.tap_layers(),
    );
    // The served arm: the verify window on `steel_attention` (its causal mask
    // in-kernel). The eager arm builds a `[k + 1, kv_len]` mask and a
    // `repeat_kv` copy per round, both of a size that grows with the cache.
    cfg.attn_impl = lloom::models::lfm2::AttnImpl::Sdpa;
    let model = build::<RewindableExact>(&cfg, &dev)?;
    let (prompt1, full2) = two_turns();
    let eos = vec![7u32];
    let n = 48usize;
    const WARMUP: usize = 4;
    fn per_round<S: Speculator<Probe = LayerTaps<5>>>(
        session: &mut Session<'_, Model<RewindableExact>>,
        spec: &mut S,
        prompt: &[u32],
        eos: &[u32],
        n: usize,
        dev: &Device,
    ) -> Result<Vec<(usize, u64, u64)>> {
        let mut rb = Blocking::new(eos, dev)?;
        let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
        // The phases drained per round (`phase_sync`), so a buffer a round
        // released is complete before the next asks for its size: the gate
        // is about shapes, not about the pool's deferred-release window,
        // which a tiny model's millisecond rounds race and a real one's do
        // not (`DESIGN.md` §6.3b).
        let opts = DecodeOptions {
            sample_split: true,
            ..DecodeOptions::new(n)
        };
        let out = session.decode_speculative(
            lloom::profile::RewindProof::of(),
            spec,
            &mut rb,
            &mut sampler,
            prompt,
            &opts,
            eos,
            &mut |_: &[u32]| Ok(()),
        )?;
        // Allocations with a pending twin are the pool's deferred-release
        // window — a size the round owns, released and not yet complete when
        // asked for again — not a size the round lacked; they are subtracted.
        Ok(out
            .stats
            .rounds
            .iter()
            .map(|r| {
                (
                    r.accepted,
                    r.timing.pool_allocations - r.timing.pool_allocations_pending_twin,
                    r.timing.residency_adds - r.timing.pool_allocations_pending_twin,
                )
            })
            .collect())
    }
    fn two_turn_rounds<S: Speculator<Probe = LayerTaps<5>>>(
        model: &Model<RewindableExact>,
        cfg: &Config,
        spec: &mut S,
        prompt1: &[u32],
        full2: &[u32],
        eos: &[u32],
        n: usize,
        dev: &Device,
    ) -> Result<Vec<(usize, u64, u64)>> {
        let mut session = Session::new(
            model,
            State::<RewindableExact>::new(true, DType::F32, cfg, dev)?,
        );
        let mut rounds = per_round(&mut session, spec, prompt1, eos, n, dev)?;
        let prepared = session.prepare(full2)?;
        assert!(
            matches!(prepared.action, ReuseAction::Rewound(_)),
            "{:?}",
            prepared.action
        );
        rounds.extend(per_round(
            &mut session,
            spec,
            &prepared.suffix,
            eos,
            n,
            dev,
        )?);
        Ok(rounds)
    }

    let mut inner = DSparkSpeculator::new_with_ctx(
        tiny_drafter(&dev, 11)?,
        model.draft_embedding().clone(),
        model.lm_head_linear().clone(),
        1,
        CtxKv::Paged {
            page_size: 32,
            shared: false,
        },
    )?;
    inner.warmup(&dev)?;
    let mut paged = Allocating {
        inner,
        dev: dev.clone(),
        allocate: false,
        n: 0,
        kept: Vec::new(),
    };
    let rounds = two_turn_rounds(&model, &cfg, &mut paged, &prompt1, &full2, &eos, n, &dev)?;
    assert!(
        rounds.len() > WARMUP + 4,
        "non-vacuous: {} rounds",
        rounds.len()
    );
    // Warm-up is by SHAPE: the observe keys `accepted` rows, so the first
    // round at each accepted length asks the pool for sizes it has not seen.
    // Every later round at a length already seen must allocate nothing.
    let mut seen = std::collections::HashSet::new();
    let mut first_seen = 0usize;
    let mut allocating = Vec::new();
    for (i, &(accepted, allocs, adds)) in rounds.iter().enumerate() {
        if seen.insert(accepted) {
            first_seen += 1;
            continue;
        }
        if allocs > 0 || adds > 0 {
            allocating.push((i, accepted, allocs, adds));
        }
    }
    assert!(
        allocating.is_empty(),
        "paged context: {} of {} rounds at an accepted length already seen allocated \
         (round, accepted, pool allocations, residency adds): {:?}",
        allocating.len(),
        rounds.len() - first_seen,
        allocating
    );
    assert!(
        rounds.len() - first_seen > 40,
        "non-vacuous: {} repeat-shape rounds",
        rounds.len() - first_seen
    );

    // The control: the same loop with a drafter that allocates per draft.
    let inner = DSparkSpeculator::new_with_ctx(
        tiny_drafter(&dev, 11)?,
        model.draft_embedding().clone(),
        model.lm_head_linear().clone(),
        1,
        CtxKv::Paged {
            page_size: 32,
            shared: false,
        },
    )?;
    let mut control = Allocating {
        inner,
        dev: dev.clone(),
        allocate: true,
        n: 0,
        kept: Vec::new(),
    };
    let rounds = two_turn_rounds(&model, &cfg, &mut control, &prompt1, &full2, &eos, n, &dev)?;
    let steady = &rounds[WARMUP..];
    assert!(
        steady.iter().all(|r| r.1 > 0),
        "the control allocates a new size every draft and the counter must see it: {steady:?}"
    );
    Ok(())
}

/// A checkpoint whose pass end lies past a rewind describes positions the
/// cache has since rewritten; it is forgotten, and a later request that
/// diverges inside the rewritten range restores from the new pass's
/// checkpoint, not the stale one. The reference is again a fresh prefill.
#[test]
fn a_rewound_sequence_does_not_restore_from_a_stale_checkpoint() -> Result<()> {
    let dev = Device::Cpu;
    let cfg = target_config(ConvState::Shuffle, KvAppend::InPlace, Vec::new());
    let model = build::<CrownExact>(&cfg, &dev)?;
    let prompt1: Vec<u32> = (0..24u32).map(|i| (i * 5 + 3) % 46).collect();
    let mut session = Session::new(
        &model,
        State::<CrownExact>::new(true, DType::F32, &cfg, &dev)?,
    );
    greedy_decode(&model, &mut session, &prompt1, 6)?;
    // Diverge at 22: positions 22.. are rewritten by the second turn.
    let mut full2 = prompt1[..22].to_vec();
    full2.extend_from_slice(&[30, 31, 32, 33, 34, 35, 36, 37]);
    let p2 = session.prepare(&full2)?;
    assert!(
        matches!(p2.action, ReuseAction::Rewound(_)),
        "{:?}",
        p2.action
    );
    greedy_decode(&model, &mut session, &p2.suffix, 6)?;
    // A third request diverging at 23 — inside the first pass's stale reach
    // (its end was 24) and inside the second pass's (its end is 30).
    let mut full3 = full2[..23].to_vec();
    full3.extend_from_slice(&[40, 41, 42]);
    let p3 = session.prepare(&full3)?;
    assert!(
        matches!(p3.action, ReuseAction::Rewound(_)),
        "{:?}",
        p3.action
    );
    assert_eq!(p3.reused, 23);
    let mut fresh = State::<CrownExact>::new(true, DType::F32, &cfg, &dev)?;
    model.forward(
        &Tensor::new(&full3[..23], &dev)?.unsqueeze(0)?,
        0,
        &mut fresh,
    )?;
    for l in 0..cfg.num_hidden_layers {
        let (Some(a), Some(b)) = (
            session.sequence().cache().conv_state_for_test(l),
            fresh.conv_state_for_test(l),
        ) else {
            continue;
        };
        let d = max_abs_diff(&a, &b)?;
        assert!(
            d == 0.0,
            "layer {l}: the restore read a stale checkpoint (differs from a fresh prefill by {d})"
        );
    }
    Ok(())
}

/// A request that extends the whole sequence (`Extend`: the reply kept, a new
/// user turn appended) reuses a prefix that runs THROUGH the positions the
/// rounds keyed, so the drafter's record of what it keyed must be the
/// target's token log — anchor and accepted tokens in order — or it resets.
#[test]
fn an_extended_request_keeps_the_drafters_context_through_the_reply() -> Result<()> {
    let dev = Device::Cpu;
    let dcfg = drafter_cfg();
    let cfg = target_config(
        ConvState::RotatingRing {
            k: dcfg.max_draft() + 1,
        },
        KvAppend::InPlace,
        dcfg.tap_layers(),
    );
    let model = build::<RewindableExact>(&cfg, &dev)?;
    let (prompt1, _) = two_turns();
    let eos = vec![7u32];
    let mut spec = DSparkSpeculator::new_with_ctx(
        tiny_drafter(&dev, 11)?,
        model.draft_embedding().clone(),
        model.lm_head_linear().clone(),
        1,
        CtxKv::Contiguous,
    )?;
    let mut session = Session::new(
        &model,
        State::<RewindableExact>::new(true, DType::F32, &cfg, &dev)?,
    );
    let run = |session: &mut Session<'_, Model<RewindableExact>>,
               spec: &mut DSparkSpeculator,
               prompt: &[u32]|
     -> Result<()> {
        let mut rb = Blocking::new(&eos, &dev)?;
        let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
        session.decode_speculative(
            lloom::profile::RewindProof::of(),
            spec,
            &mut rb,
            &mut sampler,
            prompt,
            &DecodeOptions::new(12),
            &eos,
            &mut |_: &[u32]| Ok(()),
        )?;
        Ok(())
    };
    run(&mut session, &mut spec, &prompt1)?;
    assert!(
        session.tokens().len() > prompt1.len() + 4,
        "the reply was kept in the log"
    );
    let mut full2 = session.tokens().to_vec();
    full2.extend_from_slice(&[21, 22, 23, 24, 25, 26]);
    let prepared = session.prepare(&full2)?;
    assert_eq!(
        prepared.action,
        ReuseAction::Extend,
        "the log is a prefix of the request"
    );
    run(&mut session, &mut spec, &prepared.suffix)?;
    assert_eq!(
        spec.ctx_resets(),
        0,
        "the drafter's keyed record is the target's log through the reply"
    );
    assert_eq!(
        spec.ctx_state(SeqId(0)),
        Some((session.pos(), session.pos()))
    );
    Ok(())
}
