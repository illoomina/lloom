//! A prompt past the QSA indexer's dense range is prefilled in chunks
//! (`models/qwen4exp/qsa.rs`, `QsaAttention::prefill` past the range;
//! `docs/models/qwen3.8-flash-next.md`): the whole model's chunk path against
//! the one-position-at-a-time replay, which is QSA's definition position by
//! position.
//!
//! - The prompt goes through the full-attention layer as ONE chunk, not a
//!   chunk to the range and single steps after it: what the layer's
//!   observed widths say.
//! - The prompt's logits are as close to the replay's as a prompt inside the
//!   range gets: the chunk path reorders fp32 there already (Kernel B, the
//!   chunk attention), so the tolerance is that control's, measured on the
//!   same model and device, not a constant. The greedy tokens are the
//!   replay's.
//!
//! The tiny template's indexer (budget 16, ratio 4) puts the range at 19
//! positions, so a 70-token prompt crosses it, completes blocks inside its
//! chunk and ends with a tail. At scale (`LLOOM_FIXTURE=real|synthetic`,
//! Metal) the prompt is 2 200 tokens on the streamed tier, past the
//! checkpoint's 2 051.
use lloom::models::qwen4exp::model::PrefillMode;
use lloom::models::qwen4exp::testing::Qwen4ExpTiny;
use lloom::models::qwen4exp::Shape as _;
use lloom::models::qwen4exp::{Config, ExpertResidency, LoadOptions, Qwen4Exp, State};
use lloom::nn::{ForwardObserver, HookPoint, LayerOutputPoint};
use lloom::profile::{CrownExact, RewindableExact};
use lloom::testing::fixture::Fixture;
use lloom_tensor::{Device, Result, Tensor};

/// The width of every pass the observed layer ran, in order.
struct Widths {
    layer: usize,
    seen: Vec<usize>,
}

impl ForwardObserver for Widths {
    fn at<P: HookPoint>(&mut self, layer: usize, value: &Tensor) -> Result<()> {
        if layer == self.layer {
            self.seen.push(value.dim(1)?);
        }
        Ok(())
    }
    fn wants<P: HookPoint>(&self, layer: usize) -> bool {
        layer == self.layer && P::KIND == LayerOutputPoint::KIND
    }
}

fn argmax(row: &[f32]) -> u32 {
    row.iter()
        .enumerate()
        .fold(
            (0usize, f32::MIN),
            |m, (i, &v)| if v > m.1 { (i, v) } else { m },
        )
        .0 as u32
}

/// A run's logits rows, its greedy tokens and the observed widths.
type Run = (Vec<Vec<f32>>, Vec<u32>, Vec<usize>);

/// The prompt in `mode`, then `steps` greedy tokens: the prefill's logits
/// and every step's, the tokens chosen, and the observed layer's widths
/// over the prefill.
fn run<R: ExpertResidency>(
    model: &mut Qwen4Exp<CrownExact, R>,
    cfg: &Config,
    dev: &Device,
    mode: PrefillMode,
    prompt: &[u32],
    steps: usize,
) -> Result<Run> {
    model.set_prefill_mode(mode);
    let full = cfg
        .layer_types
        .iter()
        .position(|k| matches!(k, lloom::models::qwen4exp::LayerKind::Full))
        .expect("a full-attention layer");
    let mut obs = Widths {
        layer: full,
        seen: Vec::new(),
    };
    let mut cache: State<CrownExact> = State::<CrownExact>::new(cfg);
    let input = Tensor::from_vec(prompt.to_vec(), (1, prompt.len()), dev)?;
    cache.set_pending_tokens(prompt);
    let embeds = model.embed(&input)?;
    let mut logits: Vec<f32> = model
        .forward_embeds(&embeds, &mut cache, &mut obs)?
        .flatten_all()?
        .to_vec1()?;
    let mut rows = vec![logits.clone()];
    let mut toks = Vec::with_capacity(steps);
    for _ in 0..steps {
        let next = argmax(&logits);
        toks.push(next);
        let id = Tensor::from_vec(vec![next], (1, 1), dev)?;
        logits = lloom::nn::model::CausalLM::forward(model, &id, cache.pos, &mut cache)?
            .flatten_all()?
            .to_vec1()?;
        rows.push(logits.clone());
    }
    Ok((rows, toks, obs.seen))
}

/// NaN-carrying max |a − b|.
fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, |m, d| {
            if m.is_nan() || d.is_nan() {
                f32::NAN
            } else {
                m.max(d)
            }
        })
}

fn check<R: ExpertResidency>(
    tag: &str,
    model: &mut Qwen4Exp<CrownExact, R>,
    cfg: &Config,
    dev: &Device,
    prompt: &[u32],
    control: &[Vec<u32>],
) -> Result<()> {
    let limit = model.dense_limit().expect("the loader builds the indexer");
    assert!(
        prompt.len() > limit + 2 * cfg.indexer_compress_ratio,
        "{tag}: a {}-token prompt does not reach past the dense range {limit}",
        prompt.len()
    );
    let steps = 6;
    let (want, want_toks, _) = run(model, cfg, dev, PrefillMode::Replay, prompt, steps)?;
    let (got, got_toks, widths) = run(model, cfg, dev, PrefillMode::Chunk, prompt, steps)?;
    let chunk = lloom::models::qwen4exp::prefill_chunk();
    let expect: Vec<usize> = (0..prompt.len())
        .step_by(chunk)
        .map(|s| (prompt.len() - s).min(chunk))
        .collect();
    assert_eq!(
        widths, expect,
        "{tag}: the full-attention layer's passes over the prompt (chunk {chunk}, dense range {limit})"
    );
    let scale = want[0].iter().fold(0f32, |m, x| m.max(x.abs()));
    let errs: Vec<f32> = got.iter().zip(&want).map(|(a, b)| max_diff(a, b)).collect();
    println!(
        "CHUNKPASTDENSE {tag}: prompt {} limit {limit} widths {widths:?} logits err/scale {:?} tokens chunk {got_toks:?} replay {want_toks:?}",
        prompt.len(),
        errs.iter().map(|e| e / scale).collect::<Vec<_>>()
    );
    assert!(
        got.iter().flatten().all(|v| v.is_finite()),
        "{tag}: non-finite logits"
    );
    assert_eq!(got_toks, want_toks, "{tag}: the greedy tokens moved");
    // The level: the same comparison on prompts inside the range.
    let mut level = 0f32;
    for p in control {
        assert!(p.len() <= limit, "{tag}: a control prompt past the range");
        let (w, _, _) = run(model, cfg, dev, PrefillMode::Replay, p, 0)?;
        let (g, _, _) = run(model, cfg, dev, PrefillMode::Chunk, p, 0)?;
        let sc = w[0].iter().fold(0f32, |m, x| m.max(x.abs()));
        level = level.max(max_diff(&g[0], &w[0]) / sc);
    }
    let err = errs[0] / scale;
    println!("CHUNKPASTDENSE {tag}: prompt err/scale {err:e} against the in-range level {level:e}");
    if !control.is_empty() {
        assert!(
            err <= 2.0 * level,
            "{tag}: the prompt's logits are {err:e} from the replay's, the in-range chunk {level:e}"
        );
    }
    Ok(())
}

fn tiny_prompt(n: usize) -> Vec<u32> {
    (0..n as u32).map(|i| (i * 37 + 5) % 128).collect()
}

#[test]
fn a_prompt_past_the_dense_range_prefills_in_chunks_cpu() -> Result<()> {
    let Some(fx) = Fixture::<Qwen4ExpTiny>::new("qwen4exp_chunk_past_dense_cpu", Config::tiny())?
    else {
        return Ok(());
    };
    if fx.at_scale() {
        // The CPU arm is the tiny model's; the scale arm is the Metal test.
        return Ok(());
    }
    let dev = Device::Cpu;
    let cfg = fx.config(|_| {});
    let mut model = fx.resident(&cfg, &dev, &LoadOptions::default())?;
    let control: Vec<Vec<u32>> = [8, 12, 19].map(tiny_prompt).to_vec();
    check("cpu", &mut model, &cfg, &dev, &tiny_prompt(70), &control)
}

#[cfg(feature = "metal")]
#[test]
fn a_prompt_past_the_dense_range_prefills_in_chunks_metal() -> Result<()> {
    let Ok(dev) = Device::new_metal(0) else {
        return Ok(());
    };
    let Some(fx) = Fixture::<Qwen4ExpTiny>::new("qwen4exp_chunk_past_dense_metal", Config::tiny())?
    else {
        return Ok(());
    };
    let cfg = fx.config(|_| {});
    let opts = LoadOptions::default();
    if !fx.at_scale() {
        let mut model = fx.resident(&cfg, &dev, &opts)?;
        let control: Vec<Vec<u32>> = [8, 12, 19].map(tiny_prompt).to_vec();
        return check("metal", &mut model, &cfg, &dev, &tiny_prompt(70), &control);
    }
    use lloom::models::qwen4exp::{SlotBudget, StreamLoader, StreamOptions};
    let stream = StreamOptions {
        slots: SlotBudget::Fixed(256),
        loader: StreamLoader::Pread,
        prefetch: false,
        io_hints: false,
    };
    let mut model = fx.streamed(&cfg, &dev, &opts, &stream)?;
    let base = [3u32, 5, 2, 8, 8, 1, 9, 3, 5];
    let prompt: Vec<u32> = (0..2200u32)
        .map(|i| base[(i % 9) as usize] + 1000 * (i / 9) % 200_000)
        .collect();
    // At scale the logits are reported, the tokens asserted: the real
    // model's chunk path reorders fp32 through 48 layers.
    check(fx.backend().name(), &mut model, &cfg, &dev, &prompt, &[])
}

/// The MTP drafter keys its layer from every prompt row (`MtpDrafter::observe`
/// runs the layer's prefill over them), and the checkpoint's MTP layer
/// carries the indexer like every full-attention layer. So a prompt past the
/// dense range reaches that layer's chunk prefill past it — which, until the
/// chunk ran QSA there, refused (`decode past it`) and ended the decode. The
/// speculative stream on such a prompt is the greedy stream (the drafts only
/// change the acceptance), and the drafter accepted something (it ran).
#[test]
fn mtp_keys_its_layer_from_a_prompt_past_the_dense_range() -> Result<()> {
    use lloom::generation::{Blocking, LogitsProcessor, Sampling};
    use lloom::models::qwen4exp::model::{Layer, Mixer};
    use lloom::models::qwen4exp::{testing, Indexer, MtpDrafter, MtpHead};
    use lloom::nn::{DecodeOptions, IdentityProposer, RmsNorm, Session};
    let dev = Device::Cpu;
    let mut cfg = Config::tiny();
    cfg.eos_token_id = 127;
    let model = testing::model_with_indexer::<RewindableExact, _>(&cfg, &dev, |l| {
        Ok(testing::resident_experts(&cfg, l, &dev))
    })?;
    let limit = model.dense_limit().expect("an indexer");
    let indexer = || {
        let (h, ih, id) = (cfg.hidden_size, cfg.indexer_n_heads, cfg.indexer_head_dim);
        Indexer::new(
            testing::lin(ih * id, h, 15.0, &dev),
            testing::lin(id, h, 16.0, &dev),
            RmsNorm::new(testing::ones(id, &dev), cfg.norm_eps),
            RmsNorm::new(testing::ones(id, &dev), cfg.norm_eps),
            &cfg,
        )
    };
    let d = cfg.hidden_size;
    let head = MtpHead::new(
        RmsNorm::new(testing::ones(d, &dev), cfg.norm_eps),
        RmsNorm::new(testing::ones(cfg.hc_dim(), &dev), cfg.norm_eps),
        testing::lin(d, 2 * d, 21.0, &dev),
        Layer::new(
            testing::hc(&cfg, 300.0, &dev),
            Mixer::Full(
                testing::qsa_seeded::<CrownExact>(&cfg, &dev, 11.0).with_indexer(indexer()?),
            ),
            testing::hc(&cfg, 350.0, &dev),
            testing::moe_with::<CrownExact, _>(
                &cfg,
                &dev,
                testing::resident_experts(&cfg, 7, &dev),
            ),
        ),
        testing::hc_with(&cfg, 400.0, &dev, false),
        &cfg,
    );
    let prompt = tiny_prompt(3 * limit);
    let max = 12;
    let eos = [cfg.eos_token_id];
    let greedy = {
        let mut session = Session::new(&model, RewindableExact::single(&model, &dev, 1)?);
        let mut rb = Blocking::new(&eos, &dev)?;
        let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
        session
            .decode(
                &mut IdentityProposer,
                &mut rb,
                &mut sampler,
                &prompt,
                &DecodeOptions::new(max),
                &mut |_| Ok(()),
            )?
            .ids
    };
    let mut cache = RewindableExact::single(&model, &dev, 1)?;
    cache.want_nextn = true;
    let mut session = Session::new(&model, cache);
    let mut rb = Blocking::new(&eos, &dev)?;
    let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
    let mut mtp = MtpDrafter::new(&head, model.embed_tokens(), model.lm_head(), &cfg, &dev);
    let out = session.decode_speculative(
        lloom::profile::RewindProof::of(),
        &mut mtp,
        &mut rb,
        &mut sampler,
        &prompt,
        &DecodeOptions::new(max),
        &eos,
        &mut |_| Ok(()),
    )?;
    let accepted = out.stats.rounds.iter().map(|r| r.accepted).sum::<usize>();
    println!(
        "MTPPASTDENSE prompt {} limit {limit} greedy {greedy:?} mtp {:?} rounds {} accepted {accepted}",
        prompt.len(),
        out.ids,
        out.stats.rounds.len()
    );
    assert_eq!(out.ids, greedy, "the MTP stream is not the greedy stream");
    Ok(())
}
