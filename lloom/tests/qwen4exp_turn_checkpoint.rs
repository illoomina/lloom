//! The turn checkpoint on Qwen4-exp's own state (`sequence-reuse-design.md`
//! §2.3b): the same ring and the same side recurrence as the generic
//! decoder's (`decoder_turn_checkpoint.rs`), hosted by `qwen4exp::State`
//! inside `prefill_mix`. Save/overwrite/restore preserves the saved state
//! bit for bit; prefix causality uses an identical pass shape. The hint lands
//! inside a pass and inside a pipelined pass's second tile, and the chat
//! re-render is `Rewound(reply + tail)` through `Session::prepare` with
//! the fresh session's stream. The tiny resident model, CPU and Metal.

use lloom::generation::{Blocking, LogitsProcessor, Sampling};
use lloom::models::qwen4exp::testing::{self, resident_model};
use lloom::models::qwen4exp::{
    Config, MmapPleTable, NgramHash, PleLayer, PleTable, Qwen4Exp, Resident, State,
};
use lloom::nn::model::{CacheRewind, CausalLM};
use lloom::nn::{DecodeOptions, IdentityProposer, ReuseAction, Session};
use lloom::profile::{CrownExact, Profile};
use lloom_tensor::quantized::gguf_file::{self, Content};
use lloom_tensor::quantized::{GgmlDType, QTensor};
use lloom_tensor::{Device, Result, Tensor};
use std::fs::File;
use std::path::Path;

fn pin() {
    let _ = lloom_tensor::tuning::pin_regime(lloom_tensor::tuning::Contract::Deterministic);
}

fn tokens(n: usize, vocab: usize, seed: u32) -> Vec<u32> {
    (0..n as u32)
        .map(|i| (i * 7 + seed) % vocab as u32)
        .collect()
}

fn tensor_diff(a: &Tensor, b: &Tensor) -> Result<(f32, f32)> {
    assert_eq!(a.dims(), b.dims(), "shapes differ");
    let a = a
        .flatten_all()?
        .to_dtype(lloom_tensor::DType::F32)?
        .to_vec1::<f32>()?;
    let b = b
        .flatten_all()?
        .to_dtype(lloom_tensor::DType::F32)?
        .to_vec1::<f32>()?;
    assert_eq!(a.len(), b.len(), "shapes differ");
    assert!(
        a.iter().chain(&b).all(|x| x.is_finite()),
        "non-finite state"
    );
    let diff = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    let scale = b.iter().map(|x| x.abs()).fold(0f32, f32::max);
    Ok((diff, scale))
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
    Ok(tensor_diff(a, b)?.0)
}

fn relative_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
    let (diff, scale) = tensor_diff(a, b)?;
    Ok(if diff == 0.0 { 0.0 } else { diff / scale })
}

// CPU pass splits change matmul dispatch. This parity bound does not assert
// split-invariant bits; Metal cross-shape differences remain diagnostic.
const PREFIX_REL: f32 = 1e-5;

fn state_diff<P: Profile>(a: &State<P>, b: &State<P>) -> Result<(f32, f32)> {
    state_diff_by(a, b, max_abs_diff)
}

/// Every layer of `a` against `b`: the recurrent channels' largest
/// difference and the KV rows' (the lengths must agree).
fn state_diff_by<P: Profile>(
    a: &State<P>,
    b: &State<P>,
    diff: fn(&Tensor, &Tensor) -> Result<f32>,
) -> Result<(f32, f32)> {
    assert_eq!(a.kv.len(), b.kv.len(), "layer count");
    assert_eq!(a.pos, b.pos, "position");
    assert_eq!(
        a.ngram.window(None),
        b.ngram.window(None),
        "the PLE's n-gram window"
    );
    let (mut rec, mut kv) = (0f32, 0f32);
    for l in 0..a.kv.len() {
        for (what, s, t) in [
            ("S", a.s_state.get(l), b.s_state.get(l)),
            ("conv", a.conv_state.get(l), b.conv_state.get(l)),
        ] {
            match (s, t) {
                (Some(s), Some(t)) => rec = rec.max(diff(s, t)?),
                (None, None) => {}
                _ => panic!("layer {l}: {what} written on one side only"),
            }
        }
        match (a.ple_conv.get(l), b.ple_conv.get(l)) {
            (Some(s), Some(t)) => rec = rec.max(diff(s, t)?),
            (None, None) => {}
            _ => panic!("layer {l}: the PLE history written on one side only"),
        }
        assert_eq!(
            a.kv[l].lens(),
            b.kv[l].lens(),
            "layer {l}: KV/indexer lengths"
        );
        for (part, (s, t)) in a.kv[l]
            .state_views()?
            .into_iter()
            .zip(b.kv[l].state_views()?)
            .enumerate()
        {
            match (s, t) {
                (Some(s), Some(t)) => kv = kv.max(diff(&s, &t)?),
                (None, None) => {}
                _ => panic!("layer {l}: KV/indexer part {part} written on one side only"),
            }
        }
    }
    Ok((rec, kv))
}

#[test]
#[should_panic(expected = "non-finite state")]
fn state_comparison_rejects_nan() {
    let a = Tensor::from_vec(vec![f32::NAN], 1, &Device::Cpu).unwrap();
    let b = Tensor::from_vec(vec![0f32], 1, &Device::Cpu).unwrap();
    let _ = max_abs_diff(&a, &b);
}

fn prefill(
    model: &Qwen4Exp<CrownExact, Resident>,
    cfg: &Config,
    dev: &Device,
    prompt: &[u32],
    hint: Option<usize>,
) -> Result<State<CrownExact>> {
    let mut st = model.new_cache(cfg, dev)?;
    if let Some(p) = hint {
        model.checkpoint_at(&mut st, p)?;
    }
    let input = Tensor::from_vec(prompt.to_vec(), (1, prompt.len()), dev)?;
    model.forward(&input, 0, &mut st)?;
    Ok(st)
}

fn logits(
    model: &Qwen4Exp<CrownExact, Resident>,
    st: &mut State<CrownExact>,
    pos: usize,
    toks: &[u32],
    dev: &Device,
) -> Result<Tensor> {
    let input = Tensor::from_vec(toks.to_vec(), (1, toks.len()), dev)?;
    model.forward(&input, pos, st)
}

/// An independent reference never restored from a checkpoint: the same prefix
/// pass stopped at its end. The subject runs a later pass before restoring, so
/// a missing copy cannot pass as a no-op restore.
fn exact_restore_case(
    model: &Qwen4Exp<CrownExact, Resident>,
    cfg: &Config,
    dev: &Device,
    keep: usize,
) -> Result<()> {
    let prompt = tokens(keep + 7, cfg.vocab_size, 3);
    let fresh = prefill(model, cfg, dev, &prompt[..keep], None)?;
    let mut st = prefill(model, cfg, dev, &prompt[..keep], Some(keep))?;
    logits(model, &mut st, keep, &prompt[keep..], dev)?;
    assert_eq!(st.pos, prompt.len());
    assert!(st.checkpoint_positions().contains(&keep));
    CacheRewind::restore(&mut st, keep)?;
    assert_eq!(
        state_diff(&st, &fresh)?,
        (0.0, 0.0),
        "exact restore at {keep}"
    );
    let a = logits(model, &mut st, keep, &prompt[keep..], dev)?;
    let mut fresh = fresh;
    let b = logits(model, &mut fresh, keep, &prompt[keep..], dev)?;
    assert_eq!(max_abs_diff(&a, &b)?, 0.0, "exact continuation at {keep}");
    eprintln!("Q4E-CKPT exact save/overwrite/restore keep={keep}: state=0 continuation=0");
    Ok(())
}

fn restore_case(
    model: &Qwen4Exp<CrownExact, Resident>,
    cfg: &Config,
    dev: &Device,
    n: usize,
    tail: usize,
) -> Result<()> {
    let prompt = tokens(n + 8, cfg.vocab_size, 3);
    let keep = n - tail;
    exact_restore_case(model, cfg, dev, keep)?;
    let mut st = prefill(model, cfg, dev, &prompt[..n], Some(keep))?;
    assert_eq!(st.pos, n);
    assert_eq!(
        st.checkpoint_positions(),
        vec![keep],
        "n={n} tail={tail}: the pass left its checkpoint at the hint"
    );
    assert!(CacheRewind::can_restore(&st, keep));
    if tail > 0 {
        assert!(!CacheRewind::can_restore(&st, n));
    }
    CacheRewind::restore(&mut st, keep)?;
    assert_eq!(st.pos, keep);
    let fresh = prefill(model, cfg, dev, &prompt[..keep], None)?;
    // Keep the pass shape identical while changing only tokens after the
    // checkpoint. A causal checkpoint cannot include any of that suffix.
    if tail > 0 {
        let mut changed = prompt[..n].to_vec();
        for t in &mut changed[keep..] {
            *t = (*t + 1) % cfg.vocab_size as u32;
        }
        assert_ne!(&changed[keep..], &prompt[keep..n]);
        let mut causal = prefill(model, cfg, dev, &changed, Some(keep))?;
        CacheRewind::restore(&mut causal, keep)?;
        assert_eq!(
            state_diff(&st, &causal)?,
            (0.0, 0.0),
            "checkpoint must not depend on the discarded suffix at {keep} of {n}"
        );
    }
    let (rec, kv) = state_diff(&st, &fresh)?;
    eprintln!(
        "Q4E-CKPT n={n} tail={tail} keep={keep} restore-vs-fresh: recurrent {rec:e} kv {kv:e}"
    );
    // Preserve the existing exact Metal fixtures, without claiming that all
    // shorter passes use the same arithmetic. Exact restore is tested above.
    let exact = n == keep || (dev.is_metal() && tail <= 2);
    if exact {
        assert_eq!(
            rec, 0.0,
            "n={n} tail={tail}: the restored recurrent state is a fresh prefill's"
        );
        assert_eq!(
            kv, 0.0,
            "n={n} tail={tail}: the KV rows kept are the fresh prefill's"
        );
    }
    let (rec_rel, kv_rel) = state_diff_by(&st, &fresh, relative_diff)?;
    eprintln!("Q4E-CKPT n={n} tail={tail} relative: recurrent {rec_rel:e} kv {kv_rel:e}");
    if dev.is_cpu() {
        assert!(
            rec_rel <= PREFIX_REL && kv_rel <= PREFIX_REL,
            "prefix state: recurrent {rec_rel:e}, KV {kv_rel:e}, bound {PREFIX_REL:e}"
        );
    }
    let mut fresh = fresh;
    let cont = &prompt[keep..n + 6];
    let a = logits(model, &mut st, keep, cont, dev)?;
    let b = logits(model, &mut fresh, keep, cont, dev)?;
    let d = max_abs_diff(&a, &b)?;
    eprintln!("Q4E-CKPT n={n} tail={tail} continuation: {d:e}");
    if dev.is_cpu() {
        assert!(
            relative_diff(&a, &b)? <= PREFIX_REL,
            "fresh-prefix continuation"
        );
    }
    if exact {
        assert_eq!(
            d, 0.0,
            "the continuation from the restored state is the fresh state's"
        );
    }
    Ok(())
}

/// The control: a pass with no hint checkpoints its end; restored there (a
/// no-op) the state is not a two-pass state ending at `n − tail`.
fn wrong_checkpoint_differs(
    model: &Qwen4Exp<CrownExact, Resident>,
    cfg: &Config,
    dev: &Device,
    n: usize,
    tail: usize,
) -> Result<()> {
    let prompt = tokens(n, cfg.vocab_size, 3);
    let mut st = prefill(model, cfg, dev, &prompt, None)?;
    assert_eq!(st.checkpoint_positions(), vec![n]);
    assert!(!CacheRewind::can_restore(&st, n - tail));
    assert!(CacheRewind::restore(&mut st, n - tail).is_err());
    CacheRewind::restore(&mut st, n)?;
    let mut two = prefill(model, cfg, dev, &prompt[..n - tail], None)?;
    logits(model, &mut two, n - tail, &prompt[n - tail..], dev)?;
    let (rec, _) = state_diff(&st, &two)?;
    eprintln!("Q4E-CKPT control n={n} tail={tail}: one pass vs two passes differ by {rec:e}");
    assert!(rec > 0.0, "the control: the comparison can fail");
    Ok(())
}

/// A PLE table in a GGUF, as the loader names it (`qwen4exp_host_feed.rs`).
fn write_table(cfg: &Config, rows: usize, path: &Path) -> Result<()> {
    let table = testing::wt((rows, cfg.ple_head_dim), 3.0, &Device::Cpu);
    let q = QTensor::quantize(&table, GgmlDType::F32)?;
    let mut f = File::create(path)?;
    gguf_file::write(&mut f, &[], &[("per_layer_token_embd.weight", &q)])?;
    let _ = Content::read(&mut File::open(path)?)?;
    Ok(())
}

/// The tiny resident model with the PLE at layer 1 (the served Flash-Next
/// loads its PLE: its conv history and n-gram window are checkpointed).
fn model_with_ple(cfg: &Config, dev: &Device) -> Result<Qwen4Exp<CrownExact, Resident>> {
    let path = std::env::temp_dir().join(format!(
        "lloom-q4e-turn-ckpt-{}-{}.gguf",
        std::process::id(),
        dev.is_metal()
    ));
    write_table(cfg, 72, &path)?;
    let mut model = resident_model::<CrownExact>(cfg, dev);
    let hash = NgramHash::new(
        vec![1_000_003, 999_983, 7],
        vec![13, 17, 19, 23],
        vec![0, 13, 30, 49],
        2,
        cfg.eos_token_id,
    )?;
    let hc_dim = cfg.hc_dim();
    let ple_dim = cfg.ple_head_dim * hash.n_heads();
    let block = PleLayer::<CrownExact>::new(
        testing::lin(hc_dim, ple_dim, 21.0, dev),
        testing::lin(cfg.hidden_size, ple_dim, 22.0, dev),
        testing::ones(hc_dim, dev),
        testing::ones(hc_dim, dev),
        testing::ones(hc_dim, dev),
        testing::wt((hc_dim, 4), 23.0, dev),
        cfg.hc_count,
        cfg.hidden_size,
        cfg.ple_ngram_size,
        cfg.norm_eps as f32,
    )?;
    model.set_ple(1, hash, PleTable::Mmap(MmapPleTable::open(&path)?), block)?;
    Ok(model)
}

fn cases(dev: &Device) -> Result<()> {
    let cfg = Config::tiny();
    // The PLE's state restores with the rest.
    let mut ple = model_with_ple(&cfg, dev)?;
    ple.set_turn_checkpoints(2);
    for (n, tail) in [(70, 0), (70, 1), (70, 5), (70, 20), (12, 1)] {
        restore_case(&ple, &cfg, dev, n, tail)?;
    }
    wrong_checkpoint_differs(&ple, &cfg, dev, 70, 1)?;
    let mut model = resident_model::<CrownExact>(&cfg, dev);
    model.set_turn_checkpoints(2);
    for (n, tail) in [
        (70, 0),
        (70, 1),
        (70, 2),
        (70, 9),
        (70, 20),
        (70, 62),
        (12, 1),
        (7, 2),
    ] {
        restore_case(&model, &cfg, dev, n, tail)?;
    }
    wrong_checkpoint_differs(&model, &cfg, dev, 70, 1)?;
    // The hint inside the SECOND tile of a pipelined pass: the chunk at 16
    // and two tiles a pass, the hint at 30 of 40 (`prefill_mix` sees the
    // tiles in order per layer).
    let was_chunk = lloom::models::qwen4exp::prefill_chunk();
    let was_tiles = lloom::models::qwen4exp::prefill_tiles();
    lloom::models::qwen4exp::set_prefill_chunk(16);
    lloom::models::qwen4exp::set_prefill_tiles(2);
    let r = (|| -> Result<()> {
        let prompt = tokens(48, cfg.vocab_size, 3);
        let mut st = prefill(&model, &cfg, dev, &prompt[..40], Some(30))?;
        assert_eq!(st.checkpoint_positions(), vec![30]);
        CacheRewind::restore(&mut st, 30)?;
        let fresh = prefill(&model, &cfg, dev, &prompt[..30], None)?;
        let (rec, kv) = state_diff(&st, &fresh)?;
        eprintln!(
            "Q4E-CKPT pipelined tiles, hint at 30 of [16, 16, 8]: recurrent {rec:e} kv {kv:e}"
        );
        // [16, 14] and [16, 16]'s prefix have different pass shapes.
        let (rec_rel, kv_rel) = state_diff_by(&st, &fresh, relative_diff)?;
        if dev.is_cpu() {
            assert!(
                rec_rel <= PREFIX_REL && kv_rel <= PREFIX_REL,
                "pipelined prefix: recurrent {rec_rel:e}, KV {kv_rel:e}"
            );
        } else {
            assert_eq!(kv, 0.0, "the existing Metal pipelined KV fixture");
        }
        // The same partition, bit for bit: a pass of [16, 16] cut to 30 is
        // the hinted pass itself.
        let mut same = prefill(&model, &cfg, dev, &prompt[..32], Some(30))?;
        CacheRewind::restore(&mut same, 30)?;
        assert_eq!(state_diff(&st, &same)?, (0.0, 0.0));
        Ok(())
    })();
    lloom::models::qwen4exp::set_prefill_chunk(was_chunk);
    lloom::models::qwen4exp::set_prefill_tiles(was_tiles);
    r
}

#[test]
fn a_restore_is_a_fresh_prefill_of_the_kept_prefix_on_the_cpu() -> Result<()> {
    pin();
    cases(&Device::Cpu)
}

#[cfg(feature = "metal")]
#[test]
fn a_restore_is_a_fresh_prefill_of_the_kept_prefix_on_metal() -> Result<()> {
    pin();
    cases(&Device::new_metal(0)?)
}

fn greedy_decode(
    session: &mut Session<'_, Qwen4Exp<CrownExact, Resident>>,
    prompt: &[u32],
    n: usize,
    dev: &Device,
) -> Result<Vec<u32>> {
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

/// The chat re-render through `Session::prepare` (turn_checkpoint.rs's
/// shape): `Rewound(reply + tail)` to the hinted checkpoint, the stream
/// a fresh session's; `Fresh` with no checkpoint at the divergence.
#[test]
fn a_chat_re_render_restores_through_prepare() -> Result<()> {
    pin();
    let dev = Device::Cpu;
    let cfg = Config::tiny();
    let mut model = resident_model::<CrownExact>(&cfg, &dev);
    model.set_turn_checkpoints(2);
    // (divergence back from prompt 1's end, checkpoint back from it): the
    // checkpoint at the divergence; at the generation prompt's start before
    // it (Flash-Next's thinking-on render keeps 4 of its 5 tokens); none.
    for (tail, hint) in [
        (1, Some(1)),
        (0, Some(0)),
        (2, Some(2)),
        (1, Some(5)),
        (1, None),
    ] {
        let hinted = hint.is_some();
        let prompt1 = tokens(24, cfg.vocab_size, 3);
        let div = prompt1.len() - tail;
        let keep = prompt1.len() - hint.unwrap_or(tail);
        let mut session = Session::new(&model, model.new_cache(&cfg, &dev)?);
        if hinted {
            model.checkpoint_at(session.cache_mut(), keep)?;
        }
        let reply = greedy_decode(&mut session, &prompt1, 8, &dev)?;
        assert_eq!(reply.len(), 8);
        let before = session.tokens().len();
        let mut full2 = prompt1[..div].to_vec();
        full2.extend_from_slice(&[11, 12, 13, 14, 2, 21, 22, 23, 24, 25, 26]);
        let prepared = session.prepare(&full2)?;
        if !hinted {
            // The standing window undoes the last decode step only; the
            // divergence is `reply + 1` back.
            assert_eq!(prepared.action, ReuseAction::Fresh, "tail {tail}");
            continue;
        }
        assert_eq!(
            prepared.action,
            ReuseAction::Rewound(before - keep),
            "tail {tail}"
        );
        assert_eq!(prepared.reused, keep);
        assert_eq!(session.pos(), keep);
        let got = greedy_decode(&mut session, &prepared.suffix, 12, &dev)?;
        let mut fresh = Session::new(&model, model.new_cache(&cfg, &dev)?);
        let want = greedy_decode(&mut fresh, &full2, 12, &dev)?;
        assert_eq!(
            got, want,
            "tail {tail}: the reused turn's stream is the fresh turn's"
        );
    }
    Ok(())
}
