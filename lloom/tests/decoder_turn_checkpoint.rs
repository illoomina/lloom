//! The turn checkpoint on the generic decoder (`sequence-reuse-design.md`
//! §2.3b): a multi-token pass leaves the state at a hinted position — the
//! chat re-render's divergence, a few tokens before the prompt's end — and
//! `restore` puts every layer back there: the KV to its length, the Gated
//! DeltaNet's `S` and conv window to the copy the pass left, bit for bit
//! what a fresh prefill of exactly those tokens leaves (the recurrence run
//! over the prefix from the same `S`; the pass itself is unchanged).
//! Through `Session::prepare`, the re-render is `Rewound(reply + tail)` and
//! the reused turn's tokens are a fresh session's.
//!
//! The tiny Qwen3.5 (dense and MoE) on the CPU, and the dense one on Metal
//! for the fused prefill. The reference in every test is a different
//! mechanism than the subject (CONTRIBUTING §3.1b): a fresh prefill of the
//! kept prefix against the restored state; a fresh session's stream against
//! the reused one; and the negative control — a checkpoint at another
//! position — shows the comparison can fail.

use lloom::generation::{Blocking, LogitsProcessor, Sampling};
use lloom::models::qwen3_5::testing::{Qwen3_5MoeTiny, Qwen3_5Tiny};
use lloom::models::qwen3_5::Config;
use lloom::models::qwen4exp::{GdnState, KvCache};
use lloom::nn::decoder::{
    DecodeRecovery, DecoderState, HybridState, Layer, PreNormState, StateCapacity,
};
use lloom::nn::model::{CacheReset, CacheRewind, CausalLM};
use lloom::nn::{DecodeOptions, IdentityProposer, ReuseAction, Session};
use lloom::profile::{CrownExact, Profile, RewindableExact};
use lloom::testing::tiny::{TinyCheckpoint, TinyFamily, Weights};
use lloom_tensor::{Device, Result, Tensor};

/// Both tiny variants' layer state: the hybrid mixer's, no FFN state.
type LayerState = PreNormState<HybridState<GdnState, KvCache>, ()>;

fn pin() {
    let _ = lloom_tensor::tuning::pin_regime(lloom_tensor::tuning::Contract::Deterministic);
}

fn tokens(n: usize, vocab: usize, seed: u32) -> Vec<u32> {
    (0..n as u32)
        .map(|i| (i * 7 + seed) % vocab as u32)
        .collect()
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
    let a = a
        .flatten_all()?
        .to_dtype(lloom_tensor::DType::F32)?
        .to_vec1::<f32>()?;
    let b = b
        .flatten_all()?
        .to_dtype(lloom_tensor::DType::F32)?
        .to_vec1::<f32>()?;
    assert_eq!(a.len(), b.len(), "shapes differ");
    Ok(a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max))
}

/// Every layer of `a` against `b`: the largest difference over the recurrent
/// layers' `S` and conv windows, and over the attention layers' live KV
/// rows (their lengths must agree).
fn state_diff<L: Layer<State = LayerState>>(
    a: &DecoderState<L>,
    b: &DecoderState<L>,
) -> Result<(f32, f32)> {
    assert_eq!(a.pos, b.pos, "position");
    assert_eq!(a.layers.len(), b.layers.len());
    let (mut rec, mut kv) = (0f32, 0f32);
    for (l, (x, y)) in a.layers.iter().zip(&b.layers).enumerate() {
        match (&x.mixer, &y.mixer) {
            (HybridState::A(p), HybridState::A(q)) => {
                for (what, s, t) in [
                    ("S", p.s.get(p.layer), q.s.get(q.layer)),
                    ("conv", p.conv.get(p.layer), q.conv.get(q.layer)),
                ] {
                    let (Some(s), Some(t)) = (s, t) else {
                        panic!("layer {l}: {what} written on one side only");
                    };
                    rec = rec.max(max_abs_diff(s, t)?);
                }
            }
            (HybridState::B(p), HybridState::B(q)) => {
                assert_eq!(p.len(), q.len(), "layer {l}: KV length");
                let n = p.len();
                let [pk, pv, _, _] = p.state_views()?;
                let [qk, qv, _, _] = q.state_views()?;
                for (s, t) in [(pk, qk), (pv, qv)] {
                    let (Some(s), Some(t)) = (s, t) else { continue };
                    if s.rank() == 3 && t.rank() == 3 && s.dim(1)? >= n && t.dim(1)? >= n {
                        kv = kv.max(max_abs_diff(&s.narrow(1, 0, n)?, &t.narrow(1, 0, n)?)?);
                    }
                }
            }
            _ => panic!("layer {l}: the two states are of different layer kinds"),
        }
    }
    Ok((rec, kv))
}

/// A fresh state with `prompt` prefilled in one forward, the checkpoint
/// hinted at `hint` (else at the pass's end).
fn prefill<M: CausalLM>(
    model: &M,
    cfg: &M::Config,
    dev: &Device,
    prompt: &[u32],
    hint: Option<usize>,
) -> Result<M::Cache> {
    let mut st = model.new_cache(cfg, dev)?;
    if let Some(p) = hint {
        model.checkpoint_at(&mut st, p)?;
    }
    let input = Tensor::from_vec(prompt.to_vec(), (1, prompt.len()), dev)?;
    model.forward(&input, 0, &mut st)?;
    Ok(st)
}

fn logits<M: CausalLM>(
    model: &M,
    st: &mut M::Cache,
    pos: usize,
    toks: &[u32],
    dev: &Device,
) -> Result<Tensor> {
    let input = Tensor::from_vec(toks.to_vec(), (1, toks.len()), dev)?;
    model.forward(&input, pos, st)
}

/// The fewest rows a prefill runs through the same kernel arms as a 70-row
/// one: below it a fresh prefill of the kept prefix takes the decode step
/// (one row, every backend) or another GEMM tile (Metal: the 16-row tile
/// under `SKINNY_TILE_MAX_M` = 32, `mlx_gemm::select_tile_config`, and a
/// guarded variant at `m` = 32 — measured: 8 and 32 differ, 50 and up
/// agree), and its bits are that arm's — the restored state is still the
/// pass's own prefix, but the reference is a different computation
/// (`porting-models.md` §21.10, "a different kernel arm"). Such cases are
/// printed, not asserted.
fn same_arms_from(dev: &Device) -> usize {
    if dev.is_metal() {
        48
    } else {
        2
    }
}

/// Prefill `n` tokens with the checkpoint hinted `tail` before the end,
/// restore there, and compare against a fresh prefill of the first `n −
/// tail` tokens: the states bit for bit (from [`same_arms_from`] rows),
/// then the same continuation's logits. Returns the recurrent and KV
/// state differences for the caller's record.
fn restore_case<M, L>(
    model: &M,
    cfg: &Config,
    dev: &Device,
    n: usize,
    tail: usize,
) -> Result<(f32, f32)>
where
    M: CausalLM<Config = Config, Cache = DecoderState<L>>,
    L: Layer<State = LayerState>,
{
    let prompt = tokens(n + 8, cfg.vocab_size, 3);
    let keep = n - tail;
    let mut st = prefill(model, cfg, dev, &prompt[..n], Some(keep))?;
    assert_eq!(st.pos, n);
    assert_eq!(
        st.checkpoint_positions(),
        vec![keep],
        "n={n} tail={tail}: the pass left its checkpoint at the hint"
    );
    assert!(st.can_restore(keep));
    assert!(!st.can_restore(keep.wrapping_sub(1)));
    if tail > 0 {
        assert!(
            !st.can_restore(n),
            "no checkpoint stands at the pass's end when one was hinted inside it"
        );
    }
    st.restore(keep)?;
    assert_eq!(st.pos, keep);
    let fresh = prefill(model, cfg, dev, &prompt[..keep], None)?;
    let (rec, kv) = state_diff(&st, &fresh)?;
    eprintln!(
        "TURN-CKPT n={n} tail={tail} keep={keep} restore-vs-fresh: recurrent {rec:e} kv {kv:e}"
    );
    let exact = keep >= same_arms_from(dev);
    if exact {
        assert_eq!(
            rec, 0.0,
            "n={n} tail={tail}: a restored recurrent state is a fresh prefill's, bit for bit"
        );
        assert_eq!(
            kv, 0.0,
            "n={n} tail={tail}: the KV rows kept are the fresh prefill's"
        );
    }
    // The same continuation from the two states: a multi-token pass of the
    // dropped tail and more, then two decode steps.
    let mut fresh = fresh;
    let cont = &prompt[keep..n + 6];
    let a = logits(model, &mut st, keep, cont, dev)?;
    let b = logits(model, &mut fresh, keep, cont, dev)?;
    let d1 = max_abs_diff(&a, &b)?;
    let (mut da, mut db) = (0f32, 0f32);
    for (i, t) in prompt[n + 6..n + 8].iter().enumerate() {
        let pos = n + 6 + i;
        let a = logits(model, &mut st, pos, &[*t], dev)?;
        let b = logits(model, &mut fresh, pos, &[*t], dev)?;
        da = da.max(max_abs_diff(&a, &b)?);
        db = db.max(a.abs()?.max_all()?.to_scalar::<f32>()?);
    }
    eprintln!(
        "TURN-CKPT n={n} tail={tail} continuation: pass {d1:e} steps {da:e} (|logits| ≤ {db:e})"
    );
    if exact {
        assert_eq!(
            d1, 0.0,
            "the continuation pass from the restored state is the fresh state's"
        );
        assert_eq!(da, 0.0, "the decode steps after it agree");
    }
    Ok((rec, kv))
}

/// The negative control: the comparison fails on the wrong checkpoint. A
/// pass with no hint leaves its checkpoint at its end; restored there (a
/// no-op) the state is not a fresh prefill of `n − tail` tokens.
fn wrong_checkpoint_differs<M, L>(
    model: &M,
    cfg: &Config,
    dev: &Device,
    n: usize,
    tail: usize,
) -> Result<()>
where
    M: CausalLM<Config = Config, Cache = DecoderState<L>>,
    L: Layer<State = LayerState>,
{
    let prompt = tokens(n, cfg.vocab_size, 3);
    let mut st = prefill(model, cfg, dev, &prompt, None)?;
    assert_eq!(
        st.checkpoint_positions(),
        vec![n],
        "a pass with no hint checkpoints its end"
    );
    assert!(!st.can_restore(n - tail));
    assert!(
        st.restore(n - tail).is_err(),
        "no checkpoint stands at {}",
        n - tail
    );
    st.restore(n)?;
    let mut fresh = prefill(model, cfg, dev, &prompt[..n - tail], None)?;
    // Same position for the comparison: the fresh state fed the tail as
    // its own pass (a different chunking than the one pass — the
    // recurrence's bits differ, and the control is that they do).
    logits(model, &mut fresh, n - tail, &prompt[n - tail..], dev)?;
    let (rec, _) = state_diff(&st, &fresh)?;
    eprintln!("TURN-CKPT control n={n} tail={tail}: one pass vs two passes differ by {rec:e}");
    assert!(
        rec > 0.0,
        "the control: a state the pass left differs from a two-pass one"
    );
    Ok(())
}

fn dense<P: Profile>(
    dev: &Device,
    window: usize,
) -> Result<(lloom::models::qwen3_5::Qwen3_5<P>, Config)> {
    let ckpt = TinyCheckpoint::<Qwen3_5Tiny>::new(Config::tiny(), Weights::default());
    let mut model = ckpt.model::<P>(dev)?;
    let cap = model.capacity();
    model.set_capacity(StateCapacity {
        checkpoints: 2,
        verify: window,
        recovery: if window >= 1 {
            DecodeRecovery::Standing
        } else {
            DecodeRecovery::None
        },
        ..cap
    });
    Ok((model, ckpt.cfg.clone()))
}

fn moe<P: Profile>(dev: &Device) -> Result<(lloom::models::qwen3_5::Qwen3_5Moe<P>, Config)> {
    let ckpt =
        TinyCheckpoint::<Qwen3_5MoeTiny>::new(Qwen3_5MoeTiny::template(), Weights::default());
    let mut model = ckpt.model::<P>(dev)?;
    let cap = model.capacity();
    model.set_capacity(StateCapacity {
        checkpoints: 2,
        ..cap
    });
    Ok((model, ckpt.cfg.clone()))
}

/// Both variants on the CPU: the hint at the pass's end, a few tokens
/// before it (the think tails: 2 and 4), inside the second DeltaNet chunk
/// and inside the first, and one position in.
#[test]
fn a_restore_is_a_fresh_prefill_of_the_kept_prefix_on_the_cpu() -> Result<()> {
    pin();
    let dev = Device::Cpu;
    let (model, cfg) = dense::<CrownExact>(&dev, 0)?;
    for (n, tail) in [
        (70, 0),
        (70, 2),
        (70, 4),
        (70, 9),
        (70, 62),
        (70, 66),
        (70, 69),
        (12, 2),
        (3, 1),
    ] {
        restore_case(&model, &cfg, &dev, n, tail)?;
    }
    wrong_checkpoint_differs(&model, &cfg, &dev, 70, 2)?;
    let (model, cfg) = moe::<CrownExact>(&dev)?;
    for (n, tail) in [(70, 2), (7, 4)] {
        restore_case(&model, &cfg, &dev, n, tail)?;
    }
    wrong_checkpoint_differs(&model, &cfg, &dev, 70, 2)?;
    Ok(())
}

/// The hint inside an earlier chunk than the pass's last (the decoder's
/// internal prefill chunk at 16, the hint at 50 of 70), and a pass split
/// into two forwards with the hint in the first (a tiled prefill).
#[test]
fn a_hint_inside_an_earlier_chunk_or_tile_is_met() -> Result<()> {
    pin();
    let dev = Device::Cpu;
    let (mut model, cfg) = dense::<CrownExact>(&dev, 0)?;
    model.set_prefill_chunk(16);
    restore_case(&model, &cfg, &dev, 70, 20)?;
    restore_case(&model, &cfg, &dev, 70, 2)?;
    model.set_prefill_chunk(512);
    let prompt = tokens(70, cfg.vocab_size, 3);
    let mut st = model.new_cache(&cfg, &dev)?;
    model.checkpoint_at(&mut st, 38)?;
    logits(&model, &mut st, 0, &prompt[..40], &dev)?;
    assert_eq!(st.checkpoint_positions(), vec![38]);
    logits(&model, &mut st, 40, &prompt[40..70], &dev)?;
    assert_eq!(
        st.checkpoint_positions(),
        vec![38],
        "a hinted state takes no checkpoint of its own: the second tile leaves none"
    );
    st.restore(38)?;
    let fresh = prefill(&model, &cfg, &dev, &prompt[..38], None)?;
    let (rec, kv) = state_diff(&st, &fresh)?;
    assert_eq!((rec, kv), (0.0, 0.0));
    // The ring holds two; a hint at a position a checkpoint stands at is
    // met already; the oldest slot is reused after that.
    model.checkpoint_at(&mut st, 38)?;
    logits(&model, &mut st, 38, &prompt[38..50], &dev)?;
    assert_eq!(st.checkpoint_positions(), vec![38]);
    model.checkpoint_at(&mut st, 60)?;
    logits(&model, &mut st, 50, &prompt[50..62], &dev)?;
    let mut have = st.checkpoint_positions();
    have.sort();
    assert_eq!(have, vec![38, 60]);
    model.checkpoint_at(&mut st, 65)?;
    logits(&model, &mut st, 62, &prompt[62..70], &dev)?;
    let mut have = st.checkpoint_positions();
    have.sort();
    assert_eq!(
        have,
        vec![60, 65],
        "the third checkpoint took the oldest slot"
    );
    st.restore(60)?;
    assert_eq!(
        st.checkpoint_positions(),
        vec![60],
        "the one past the restore is forgotten"
    );
    // The reference is the sequence's own history — the passes it ran,
    // ending at 60 — not one pass over 60 tokens: the chunked recurrence
    // sums per chunk from each pass's start, so a different pass partition
    // is a different (legitimately reordered) computation. Measured here
    // and recorded: one pass [0, 60) against these three differs.
    let mut same = prefill(&model, &cfg, &dev, &prompt[..38], None)?;
    logits(&model, &mut same, 38, &prompt[38..50], &dev)?;
    logits(&model, &mut same, 50, &prompt[50..60], &dev)?;
    assert_eq!(
        state_diff(&st, &same)?,
        (0.0, 0.0),
        "the restored state is the history's own at 60"
    );
    let one = prefill(&model, &cfg, &dev, &prompt[..60], None)?;
    let (rec, kv) = state_diff(&st, &one)?;
    eprintln!("TURN-CKPT three passes to 60 vs one pass: recurrent {rec:e} kv {kv:e}");
    assert_eq!(kv, 0.0, "the KV rows do not depend on the pass partition");
    Ok(())
}

/// The `Ring` profile with a verify window: the recurrent state on phase
/// rings, a checkpoint copied out of the current phase and restored into
/// the next; and no restore under an open window.
#[test]
fn the_ring_profile_restores_into_the_next_phase() -> Result<()> {
    pin();
    let dev = Device::Cpu;
    let (model, cfg) = dense::<RewindableExact>(&dev, 2)?;
    restore_case(&model, &cfg, &dev, 70, 2)?;
    let prompt = tokens(24, cfg.vocab_size, 3);
    let mut st = prefill(&model, &cfg, &dev, &prompt, Some(22))?;
    st.open_window()?;
    assert!(!st.can_restore(22), "no restore under an open window");
    assert!(st.restore(22).is_err());
    st.resolve_window()?;
    assert!(st.can_restore(22));
    st.reset();
    assert!(
        st.checkpoint_positions().is_empty(),
        "a reset forgets every checkpoint"
    );
    Ok(())
}

fn greedy_decode<M: CausalLM>(
    session: &mut Session<'_, M>,
    prompt: &[u32],
    n: usize,
    dev: &Device,
) -> Result<Vec<u32>>
where
    M::Cache: CacheRewind,
{
    let eos = vec![u32::MAX];
    let mut rb = Blocking::new(&eos, dev)?;
    let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
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

/// The chat shape through `Session::prepare`: turn 1's prompt ends in a
/// `tail` the re-render drops, a reply follows, turn 2 is the prompt
/// without the tail, the reply, a new turn. With the checkpoint hinted
/// `tail` before the end, turn 2 is `Rewound(reply + tail)` to the
/// checkpoint and its stream is a fresh session's; without a checkpoint
/// there it is `Fresh`.
fn re_render_case<M, L>(
    model: &M,
    cfg: &Config,
    dev: &Device,
    tail: usize,
    hinted: bool,
) -> Result<()>
where
    M: CausalLM<Config = Config, Cache = DecoderState<L>>,
    L: Layer<State = LayerState>,
    M::Cache: CacheReset,
{
    re_render_at(model, cfg, dev, tail, hinted.then_some(tail))
}

/// [`re_render_case`] with the checkpoint hinted `hint` positions before
/// prompt 1's end (the generation prompt's start) and the re-render
/// diverging `tail` before it: `prepare` restores to the latest checkpoint
/// at or before the divergence (`CacheRewind::restore_point`).
fn re_render_at<M, L>(
    model: &M,
    cfg: &Config,
    dev: &Device,
    tail: usize,
    hint: Option<usize>,
) -> Result<()>
where
    M: CausalLM<Config = Config, Cache = DecoderState<L>>,
    L: Layer<State = LayerState>,
    M::Cache: CacheReset,
{
    let prompt1 = tokens(24, cfg.vocab_size, 3);
    let div = prompt1.len() - tail;
    let hinted = hint.is_some();
    let keep = prompt1.len() - hint.unwrap_or(tail);
    let mut session = Session::new(model, model.new_cache(cfg, dev)?);
    if hinted {
        model.checkpoint_at(session.cache_mut(), keep)?;
    }
    let reply = greedy_decode(&mut session, &prompt1, 8, dev)?;
    assert_eq!(reply.len(), 8);
    let before = session.tokens().len();
    let mut full2 = prompt1[..div].to_vec();
    full2.extend_from_slice(&[11, 12, 13, 14, 2, 21, 22, 23, 24, 25, 26]);
    let prepared = session.prepare(&full2)?;
    if !hinted {
        assert_eq!(
            prepared.action,
            ReuseAction::Fresh,
            "tail {tail}: no checkpoint reaches the divergence"
        );
        return Ok(());
    }
    assert_eq!(
        prepared.action,
        ReuseAction::Rewound(before - keep),
        "tail {tail}: the re-render is a restore to the checkpoint"
    );
    assert_eq!(prepared.reused, keep);
    assert_eq!(prepared.suffix, full2[keep..]);
    assert_eq!(session.pos(), keep);
    assert_eq!(session.tokens(), &full2[..keep]);
    let got = greedy_decode(&mut session, &prepared.suffix, 12, dev)?;
    let mut fresh = Session::new(model, model.new_cache(cfg, dev)?);
    let want = greedy_decode(&mut fresh, &full2, 12, dev)?;
    assert_eq!(
        got, want,
        "tail {tail}: the reused turn's stream is the fresh turn's"
    );
    Ok(())
}

#[test]
fn a_chat_re_render_restores_through_prepare() -> Result<()> {
    pin();
    let dev = Device::Cpu;
    let (model, cfg) = dense::<CrownExact>(&dev, 0)?;
    for tail in [0, 2, 4] {
        re_render_case(&model, &cfg, &dev, tail, true)?;
    }
    re_render_case(&model, &cfg, &dev, 2, false)?;
    // The checkpoint at the generation prompt's start (5 back), the
    // re-render keeping 3 of its tokens (diverging 2 back).
    re_render_at(&model, &cfg, &dev, 2, Some(5))?;
    let (model, cfg) = dense::<RewindableExact>(&dev, 2)?;
    re_render_case(&model, &cfg, &dev, 2, true)?;
    let (model, cfg) = moe::<CrownExact>(&dev)?;
    re_render_case(&model, &cfg, &dev, 4, true)?;
    Ok(())
}

/// The fused prefill (Metal): the recurrence over the prefix through the
/// same kernels, from the same window and `S`.
#[cfg(feature = "metal")]
#[test]
fn a_restore_is_a_fresh_prefill_of_the_kept_prefix_on_metal() -> Result<()> {
    pin();
    let dev = Device::new_metal(0)?;
    let (model, cfg) = dense::<CrownExact>(&dev, 0)?;
    for (n, tail) in [
        (70, 0),
        (70, 2),
        (70, 4),
        (70, 9),
        (70, 38),
        (70, 62),
        (70, 66),
        (12, 2),
        (7, 2),
    ] {
        restore_case(&model, &cfg, &dev, n, tail)?;
    }
    wrong_checkpoint_differs(&model, &cfg, &dev, 70, 2)?;
    let (model, cfg) = dense::<RewindableExact>(&dev, 2)?;
    restore_case(&model, &cfg, &dev, 70, 2)?;
    re_render_case(&model, &cfg, &dev, 2, true)?;
    let (model, cfg) = moe::<CrownExact>(&dev)?;
    restore_case(&model, &cfg, &dev, 70, 2)?;
    Ok(())
}
