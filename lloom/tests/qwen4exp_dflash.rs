//! DFlash on the Qwen4-exp target (the hidden-state drafter family on the
//! `LayerTaps` probe): the taps a cache retains, and the speculative stream.
//!
//! - **The taps are the attention bracket's input, by layer and position.**
//!   The reference is a different code path over the same forward: an
//!   observer reading the carry entering each tapped layer (`HcStatePoint`)
//!   and that layer's bracket applied to it (`Layer::attn_input`). Bitwise,
//!   in the drafter's consumer order (a permuted order is asked for), over
//!   the chunk prefill and the decode recurrence.
//! - **A rewound step retains nothing of itself**: the standing window's
//!   rewind (a turn re-render, a redone pass) cuts the taps it kept.
//! - **The speculative stream is greedy's.** The DSpark/DFlash drafter on
//!   the target's own `token_embd`/`lm_head` (`QwenHead`), over both verify
//!   arms (one pass over the phase rings, the sequential steps), at several
//!   draft widths; an oracle (every draft accepted, the probe of every
//!   round checked against its accepted length) and a garbage drafter
//!   (every draft rejected).
//! - **The drafter config reader** on the released checkpoint, when the HF
//!   cache holds it: the keys a wrong default would silently corrupt.
//!
//! CPU: bit-stable, and the question is the bookkeeping.

use lloom::generation::{Blocking, LogitsProcessor, Sampling};
use lloom::models::lfm2_dspark::{
    drafter_params, CtxKv, DSparkConfig, DSparkDrafter, DSparkSpeculator,
};
use lloom::models::qwen4exp::model::PrefillMode;
use lloom::models::qwen4exp::{testing, Config, Qwen4Exp, QwenHead, Resident, Shape as _, State};
use lloom::nn::model::{CacheRewind, CausalLM};
use lloom::nn::observer::{ForwardObserver, HookKind, HookPoint};
use lloom::nn::speculator::{Draft, KvSource, LayerTaps, SeqId, Speculator};
use lloom::nn::{DecodeOptions, IdentityProposer, Session, VarBuilder};
use lloom::profile::{CrownExact, RewindProof, RewindableExact};
use lloom_tensor::{DType, Device, Result, Tensor};
use rand::SeedableRng;
use rand_distr::Distribution;
use std::collections::HashMap;

/// The drafter's `target_layer_ids`; the cache taps each `+ 1`.
const TARGET_IDS: [usize; 5] = [0, 1, 2, 4, 5];

fn config() -> Config {
    let mut cfg = Config::tiny();
    cfg.n_layers = 8;
    cfg.layer_types = Config::resolve_layer_types(cfg.n_layers, cfg.full_attention_interval);
    cfg.n_experts = 6;
    cfg.n_experts_per_tok = 2;
    cfg.eos_token_id = 127;
    cfg
}

fn drafter_cfg(cfg: &Config, block: usize) -> DSparkConfig {
    DSparkConfig {
        hidden_size: cfg.hidden_size,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 16,
        intermediate_size: 48,
        vocab_size: cfg.vocab_size,
        rms_norm_eps: 1e-6,
        rope_theta: 1e7,
        rope_is_neox_style: true,
        block_size: block,
        mask_token_id: 126,
        target_layer_ids: TARGET_IDS.to_vec(),
        num_target_layers: cfg.n_layers,
        markov_rank: 0,
        enable_confidence_head: false,
        logits_start: 0,
    }
}

fn randn(rng: &mut rand::rngs::StdRng, shape: Vec<usize>, dev: &Device) -> Result<Tensor> {
    let n: usize = shape.iter().product();
    let normal = rand_distr::Normal::new(0.0f32, 0.3).unwrap();
    let vs: Vec<f32> = (0..n).map(|_| normal.sample(rng)).collect();
    Tensor::from_vec(vs, shape.as_slice(), dev)
}

/// Every tensor `DSparkDrafter::new` loads for a Markov-less, confidence-less
/// drafter (the DFlash checkpoint's set), random.
fn tiny_drafter(dcfg: DSparkConfig, dev: &Device, seed: u64) -> Result<DSparkDrafter> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let h = dcfg.hidden_size;
    let hd = dcfg.head_dim;
    let q = dcfg.num_attention_heads * hd;
    let kv = dcfg.num_key_value_heads * hd;
    let i = dcfg.intermediate_size;
    let mut t: HashMap<String, Tensor> = HashMap::new();
    let mut fill = |name: String, shape: Vec<usize>| -> Result<()> {
        t.insert(name, randn(&mut rng, shape, dev)?);
        Ok(())
    };
    fill("fc.weight".into(), vec![h, dcfg.target_layer_ids.len() * h])?;
    fill("hidden_norm.weight".into(), vec![h])?;
    fill("norm.weight".into(), vec![h])?;
    for l in 0..dcfg.num_hidden_layers {
        let p = format!("layers.{l}");
        fill(format!("{p}.self_attn.q_proj.weight"), vec![q, h])?;
        fill(format!("{p}.self_attn.k_proj.weight"), vec![kv, h])?;
        fill(format!("{p}.self_attn.v_proj.weight"), vec![kv, h])?;
        fill(format!("{p}.self_attn.o_proj.weight"), vec![h, q])?;
        fill(format!("{p}.self_attn.q_norm.weight"), vec![hd])?;
        fill(format!("{p}.self_attn.k_norm.weight"), vec![hd])?;
        fill(format!("{p}.mlp.gate_proj.weight"), vec![i, h])?;
        fill(format!("{p}.mlp.up_proj.weight"), vec![i, h])?;
        fill(format!("{p}.mlp.down_proj.weight"), vec![h, i])?;
        fill(format!("{p}.input_layernorm.weight"), vec![h])?;
        fill(format!("{p}.post_attention_layernorm.weight"), vec![h])?;
    }
    let vb = VarBuilder::from_tensors(t, DType::F32, dev);
    DSparkDrafter::new(dcfg, 512, vb)
}

/// The carry entering each wanted layer, every forward, in order.
struct Carries {
    layers: Vec<usize>,
    seen: Vec<(usize, Tensor)>,
}

impl ForwardObserver for Carries {
    fn at<P: HookPoint>(&mut self, layer: usize, value: &Tensor) -> Result<()> {
        if P::KIND == HookKind::HcState {
            self.seen.push((layer, value.clone()));
        }
        Ok(())
    }
    fn wants<P: HookPoint>(&self, layer: usize) -> bool {
        P::KIND == HookKind::HcState && self.layers.contains(&layer)
    }
}

fn bits(t: &Tensor) -> Result<Vec<u32>> {
    Ok(t.flatten_all()?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?
        .into_iter()
        .map(f32::to_bits)
        .collect())
}

/// The taps a cache retains over a forward are, per tapped layer in the
/// consumer order asked for, the layer's attention-bracket input over the
/// carry the forward brought it: bitwise, under the chunk prefill and the
/// decode recurrence.
#[test]
fn the_taps_are_each_layers_attention_input_in_consumer_order() -> Result<()> {
    let cfg = config();
    let dev = Device::Cpu;
    // A permuted order: the slot a layer fills is its place in the list,
    // never its index.
    let order = [6usize, 2, 5, 1, 3];
    let prompt: Vec<u32> = (0..9).map(|i| (i * 13 + 5) % 120).collect();
    for mode in [PrefillMode::Chunk, PrefillMode::Replay] {
        let mut model: Qwen4Exp<CrownExact, Resident> = testing::resident_model(&cfg, &dev);
        model.set_prefill_mode(mode);

        let mut tapped = model.new_cache(&cfg, &dev)?;
        tapped.set_tap_layers(&order)?;
        assert_eq!(tapped.tap_layers(), order.to_vec());
        let input = Tensor::new(prompt.as_slice(), &dev)?.unsqueeze(0)?;
        model.forward(&input, 0, &mut tapped)?;
        let step = Tensor::new(&[17u32], &dev)?.unsqueeze(0)?;
        model.forward(&step, prompt.len(), &mut tapped)?;
        let taps = tapped.take_taps()?.expect("taps retained");
        assert!(tapped.take_taps()?.is_none(), "a take leaves nothing");

        let mut reference = model.new_cache(&cfg, &dev)?;
        let mut obs = Carries {
            layers: order.to_vec(),
            seen: Vec::new(),
        };
        reference.set_pending_tokens(&prompt);
        model.forward_embeds(&model.embed(&input)?, &mut reference, &mut obs)?;
        reference.set_pending_tokens(&[17]);
        model.forward_embeds(&model.embed(&step)?, &mut reference, &mut obs)?;

        let n = prompt.len() + 1;
        for (slot, &l) in order.iter().enumerate() {
            let tap = &taps[slot];
            assert_eq!(
                tap.dims(),
                &[1, n, cfg.hidden_size],
                "{mode:?} tap {slot} (layer {l})"
            );
            assert_eq!(tap.dtype(), DType::F32);
            let mut want: Vec<u32> = Vec::new();
            for (ll, carry) in obs.seen.iter().filter(|(ll, _)| *ll == l) {
                let x = model.layers()[*ll].attn_input(carry)?;
                want.extend(bits(&x)?);
            }
            assert_eq!(
                bits(tap)?,
                want,
                "{mode:?}: tap slot {slot} is not layer {l}'s attention input"
            );
        }
    }
    Ok(())
}

/// A decode step's standing window, rewound, takes the taps it retained
/// with it: what is left covers the positions the cache still holds.
#[test]
fn a_rewound_step_retains_nothing_of_itself() -> Result<()> {
    let cfg = config();
    let dev = Device::Cpu;
    let model: Qwen4Exp<CrownExact, Resident> = testing::resident_model(&cfg, &dev);
    let mut cache = model.new_cache(&cfg, &dev)?;
    cache.set_tap_layers(&[1, 3])?;
    let prompt: Vec<u32> = vec![3, 9, 27, 81, 5];
    model.forward(
        &Tensor::new(prompt.as_slice(), &dev)?.unsqueeze(0)?,
        0,
        &mut cache,
    )?;
    model.forward(&Tensor::new(&[7u32], &dev)?.unsqueeze(0)?, 5, &mut cache)?;
    assert_eq!(
        cache.rewind_horizon(),
        1,
        "a decode step leaves a standing window"
    );
    cache.rewind(1)?;
    let taps = cache.take_taps()?.expect("the prompt's taps stand");
    for t in &taps {
        assert_eq!(t.dim(1)?, prompt.len(), "the rewound step's tap was kept");
    }
    Ok(())
}

fn greedy(
    model: &Qwen4Exp<RewindableExact, Resident>,
    cfg: &Config,
    dev: &Device,
    prompt: &[u32],
    max: usize,
) -> Result<Vec<u32>> {
    let mut session = Session::new(model, RewindableExact::single(model, dev, 8)?);
    let eos = [cfg.eos_token_id];
    let mut rb = Blocking::new(&eos, dev)?;
    let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
    let out = session.decode(
        &mut IdentityProposer,
        &mut rb,
        &mut sampler,
        prompt,
        &DecodeOptions::new(max),
        &mut |_| Ok(()),
    )?;
    Ok(out.ids)
}

/// The stream, the rounds that kept a draft, and the positions the target
/// holds after the run.
fn speculative<S: Speculator<Probe = LayerTaps<5>>>(
    model: &Qwen4Exp<RewindableExact, Resident>,
    cfg: &Config,
    dev: &Device,
    spec: &mut S,
    prompt: &[u32],
    max: usize,
    ring: bool,
) -> Result<(Vec<u32>, usize, usize)> {
    let k = spec.max_draft();
    // `ring`: the verify window's phase rings sized to the drafter's `K`;
    // else a ring of 8 phases — the drivers run on the `Ring` shape alone
    // (`RewindProof`, profiles-and-memory-design.md §7.3), so the own-cache
    // arm the test once had is the same sequential verify over a wider ring.
    let mut cache: State<RewindableExact> = if ring {
        State::<RewindableExact>::in_ring(cfg, &model.spec_pools(1, k, dev))?
    } else {
        RewindableExact::single(model, dev, 8)?
    };
    let taps: Vec<usize> = TARGET_IDS.iter().map(|l| l + 1).collect();
    cache.set_tap_layers(&taps)?;
    let mut session = Session::new(model, cache);
    let eos = [cfg.eos_token_id];
    let mut rb = Blocking::new(&eos, dev)?;
    let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
    let out = session.decode_speculative(
        RewindProof::of(),
        spec,
        &mut rb,
        &mut sampler,
        prompt,
        &DecodeOptions::new(max),
        &eos,
        &mut |_| Ok(()),
    )?;
    let kept = out.stats.rounds.iter().filter(|r| r.accepted > 1).count();
    Ok((out.ids, kept, session.pos()))
}

/// A prompt whose greedy stream spans several rounds.
fn long_prompt(
    model: &Qwen4Exp<RewindableExact, Resident>,
    cfg: &Config,
    dev: &Device,
    max: usize,
) -> Result<(Vec<u32>, Vec<u32>)> {
    for seed in 0..8u32 {
        let prompt: Vec<u32> = (0..6).map(|i| (seed * 11 + i * 7 + 1) % 120).collect();
        let ids = greedy(model, cfg, dev, &prompt, max)?;
        if ids.len() >= 10 {
            return Ok((prompt, ids));
        }
    }
    lloom_tensor::bail!("no prompt gave a greedy stream of >= 10 tokens")
}

/// Proposes the greedy stream's continuation (every draft accepted) or a
/// token the target never emits (every draft rejected), and checks the
/// probe each round hands it: five taps, each covering exactly the
/// positions the round kept.
struct Scripted {
    stream: Option<Vec<u32>>,
    k: usize,
    produced: usize,
    rows: usize,
    seeded: bool,
}

impl Speculator for Scripted {
    type Probe = LayerTaps<5>;
    type Family = lloom::nn::speculator::DSpark;
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
        Ok(())
    }
    fn draft(&mut self, _: &[SeqId]) -> Result<Vec<Draft>> {
        let toks = match &self.stream {
            Some(s) => (0..self.k)
                .map(|i| s.get(self.produced + i).copied().unwrap_or(0))
                .collect(),
            None => vec![125; self.k],
        };
        Ok(vec![Draft::host(toks)])
    }
    fn observe(&mut self, _: SeqId, probe: &Vec<Tensor>, accepted: &[u32]) -> Result<()> {
        assert_eq!(probe.len(), 5, "five taps");
        let n = probe[0].dim(1)?;
        for t in probe {
            assert_eq!(t.dim(1)?, n, "every tap covers the same positions");
        }
        if self.seeded {
            assert_eq!(
                n,
                accepted.len(),
                "a round's probe covers its kept positions"
            );
        }
        self.seeded = true;
        self.rows += n;
        self.produced += accepted.len();
        Ok(())
    }
}

/// The speculative stream against greedy's on `dev`. On the CPU a verify
/// window runs position by position through the decode step; on Metal it is
/// one pass over the phase rings (`Layer::decode_parts_seq`), which retains
/// its taps on its own path — the Metal arm is what reaches it.
fn identity_on(dev: &Device) -> Result<()> {
    let cfg = config();
    let dev = dev.clone();
    let model: Qwen4Exp<RewindableExact, Resident> = testing::resident_model(&cfg, &dev);
    let max = 24;
    let (prompt, want) = long_prompt(&model, &cfg, &dev, max)?;

    for ring in [true, false] {
        // The drafter on the target's own pair, at widths below and at its block.
        for (block, k) in [(4usize, 4usize), (4, 2), (7, 7), (7, 1)] {
            let drafter = tiny_drafter(drafter_cfg(&cfg, block), &dev, 3)?;
            let mut spec = DSparkSpeculator::with_head_ctx(
                drafter,
                QwenHead::of(&model),
                1,
                CtxKv::Contiguous,
            )?;
            spec.set_max_draft(k);
            let (ids, kept, pos) = speculative(&model, &cfg, &dev, &mut spec, &prompt, max, ring)?;
            assert_eq!(
                ids, want,
                "ring={ring} block={block} k={k}: not the greedy stream"
            );
            let (ctx, at) = spec.ctx_state(SeqId(0)).expect("slot 0");
            assert_eq!(
                (ctx, at),
                (pos, pos),
                "ring={ring} block={block} k={k}: the drafter's context is not every position the \
                 target kept"
            );
            eprintln!("DFLASHID ring={ring} block={block} k={k} rounds_kept={kept}");
        }

        let mut oracle = Scripted {
            stream: Some(want.clone()),
            k: 3,
            produced: 0,
            rows: 0,
            seeded: false,
        };
        let (ids, kept, pos) = speculative(&model, &cfg, &dev, &mut oracle, &prompt, max, ring)?;
        assert_eq!(ids, want, "ring={ring}: the oracle's stream");
        assert!(kept > 0, "ring={ring}: the oracle kept no draft");
        assert_eq!(
            oracle.rows, pos,
            "ring={ring}: the oracle's probes against the target's positions"
        );

        let mut garbage = Scripted {
            stream: None,
            k: 3,
            produced: 0,
            rows: 0,
            seeded: false,
        };
        let (ids, kept, pos) = speculative(&model, &cfg, &dev, &mut garbage, &prompt, max, ring)?;
        assert_eq!(ids, want, "ring={ring}: the garbage drafter's stream");
        assert_eq!(kept, 0, "ring={ring}: a garbage draft was kept");
        assert_eq!(
            garbage.rows, pos,
            "ring={ring}: the garbage probes against the target's positions"
        );
    }
    Ok(())
}

#[test]
fn dflash_on_qwen_is_byte_identical_to_greedy() -> Result<()> {
    identity_on(&Device::Cpu)
}

#[cfg(feature = "metal")]
#[test]
fn dflash_on_qwen_is_byte_identical_to_greedy_on_metal() -> Result<()> {
    identity_on(&Device::new_metal(0)?)
}

/// Taps a forward left untaken before a verify are not the verify's: the
/// window's probe covers exactly the positions it kept, and the discarded
/// set is counted as a dropped probe (what the drivers refuse a seed over).
#[test]
fn a_verify_drops_the_taps_left_before_it() -> Result<()> {
    use lloom::nn::speculator::{Anchor, Probes};
    let cfg = config();
    let dev = Device::Cpu;
    let model: Qwen4Exp<RewindableExact, Resident> = testing::resident_model(&cfg, &dev);
    let mut cache = State::<RewindableExact>::in_ring(&cfg, &model.spec_pools(1, 2, &dev))?;
    let taps: Vec<usize> = TARGET_IDS.iter().map(|l| l + 1).collect();
    cache.set_tap_layers(&taps)?;
    let prompt: Vec<u32> = vec![3, 9, 27, 81, 5, 15];
    let logits = model.forward(
        &Tensor::new(prompt.as_slice(), &dev)?.unsqueeze(0)?,
        0,
        &mut cache,
    )?;
    let anchor = logits
        .squeeze(0)?
        .argmax(lloom_tensor::D::Minus1)?
        .to_scalar::<u32>()?;
    let drops = Probes::<LayerTaps<5>>::probe_drops(&model, &cache);
    let (accepted, probe) = Probes::<LayerTaps<5>>::verify_and_probe(
        &model,
        &[Draft::host(vec![125, 125])],
        &[Anchor {
            last: anchor,
            pos: cache.pos,
        }],
        &mut cache,
        &[cfg.eos_token_id],
    )?;
    let a = accepted[0].len();
    assert_eq!(probe.len(), 5);
    for t in &probe {
        assert_eq!(
            t.dim(1)?,
            a,
            "the probe holds the prompt's stale taps beside the window's"
        );
    }
    assert_eq!(
        Probes::<LayerTaps<5>>::probe_drops(&model, &cache),
        drops + 1,
        "the untaken prompt taps were not counted as a dropped probe"
    );
    Ok(())
}

/// A bounded run keeps nothing past its budget: every round's draft is cut
/// to what the run may still publish, the last one to the anchor alone, so
/// a cap falling mid-block leaves no overrun, the cache one position behind
/// the stream, and the next request extends the sequence rather than
/// re-prefilling it (a settled window cannot be rewound). Every budget from
/// 5 to 12 under an oracle that is always right at K = 3 puts the cap at
/// every offset of a block.
#[test]
fn a_bounded_run_keeps_nothing_past_its_budget() -> Result<()> {
    use lloom::nn::ReuseAction;
    let cfg = config();
    let dev = Device::Cpu;
    let model: Qwen4Exp<RewindableExact, Resident> = testing::resident_model(&cfg, &dev);
    let (prompt, want) = long_prompt(&model, &cfg, &dev, 24)?;
    for max in 5..=12usize {
        let k = 3;
        let mut cache = State::<RewindableExact>::in_ring(&cfg, &model.spec_pools(1, k, &dev))?;
        cache.set_tap_layers(&TARGET_IDS.iter().map(|l| l + 1).collect::<Vec<_>>())?;
        let mut session = Session::new(&model, cache);
        let mut oracle = Scripted {
            stream: Some(want.clone()),
            k,
            produced: 0,
            rows: 0,
            seeded: false,
        };
        let eos = [cfg.eos_token_id];
        let mut rb = Blocking::new(&eos, &dev)?;
        let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
        let out = session.decode_speculative(
            RewindProof::of(),
            &mut oracle,
            &mut rb,
            &mut sampler,
            &prompt,
            &DecodeOptions::new(max),
            &eos,
            &mut |_| Ok(()),
        )?;
        assert_eq!(out.ids, want[..max], "max {max}: the stream");
        assert_eq!(
            out.stats.overrun, 0,
            "max {max}: the run fed the cache past its budget"
        );
        assert_eq!(
            session.pos(),
            prompt.len() + max - 1,
            "max {max}: the cache is not one position behind the stream"
        );
        let mut next = prompt.clone();
        next.extend_from_slice(&out.ids);
        next.push(7);
        let prepared = session.prepare(&next)?;
        assert_eq!(
            prepared.action,
            ReuseAction::Extend,
            "max {max}: {prepared:?}"
        );
    }
    Ok(())
}

/// The released DFlash drafter's config (PixelML/Qwen3.8-Flash-Next-NVFP4-
/// DFlash), when the HF cache holds it: the reader takes its `rope_parameters`
/// and its class's neox rotation, and the parameter count derived from the
/// config is the header's.
#[test]
fn the_released_dflash_config_reads_as_its_checkpoint() -> Result<()> {
    let Some(home) = std::env::var_os("HOME") else {
        return Ok(());
    };
    let snaps = std::path::Path::new(&home)
        .join(".cache/huggingface/hub/models--PixelML--Qwen3.8-Flash-Next-NVFP4-DFlash/snapshots");
    let Some(dir) = std::fs::read_dir(&snaps)
        .ok()
        .and_then(|mut d| d.next())
        .and_then(|e| e.ok())
        .map(|e| e.path())
    else {
        eprintln!("SKIP: no PixelML DFlash snapshot in the HF cache");
        return Ok(());
    };
    let c = DSparkConfig::read_from_dir(&dir)?;
    assert!(
        c.rope_is_neox_style,
        "DeepSpec's Qwen3 drafter rotates neox"
    );
    assert_eq!(c.rope_theta, 1e7);
    assert_eq!(c.rms_norm_eps, 1e-6);
    assert_eq!(
        (
            c.hidden_size,
            c.num_hidden_layers,
            c.num_attention_heads,
            c.num_key_value_heads
        ),
        (2560, 5, 24, 2)
    );
    assert_eq!(
        (c.head_dim, c.intermediate_size, c.vocab_size),
        (256, 7680, 248_320)
    );
    assert_eq!(
        (c.block_size, c.mask_token_id, c.logits_start),
        (7, 248_077, 0)
    );
    assert_eq!((c.markov_rank, c.enable_confidence_head), (0, false));
    assert_eq!(c.num_target_layers, 48);
    assert_eq!(c.target_layer_ids, vec![3, 15, 23, 35, 43]);
    assert_eq!(c.tap_layers(), vec![4, 16, 24, 36, 44]);
    // Summed from the safetensors header: 58 BF16 tensors.
    assert_eq!(drafter_params(&c), 498_106_880);
    Ok(())
}

/// The sampled identity under DFlash (#943): at every cell of
/// `sampled_cells` and three seeds, the speculative run — the drafter at
/// k = 4 and 2, the sequential verify and the phase rings, the oracle on the
/// sampled stream, garbage — is the K=1 run. On Metal the ring verify is
/// one pass whose rows are an `m = k + 1` pass against the K=1 steps'
/// `m = 1`; `deterministic` promises nothing there at a near-tie, and on
/// these fixtures and seeds no row is one.
fn sampled_identity_on(dev: &Device) -> Result<()> {
    use crate::sampled_cells::{cells, k1, spec};
    let cfg = config();
    let dev = dev.clone();
    let model: Qwen4Exp<RewindableExact, Resident> = testing::resident_model(&cfg, &dev);
    let max = 20;
    let (prompt, greedy_ids) = long_prompt(&model, &cfg, &dev, max)?;
    let eos = [cfg.eos_token_id];
    let taps: Vec<usize> = TARGET_IDS.iter().map(|l| l + 1).collect();
    let cache = |ring: Option<usize>| -> Result<State<RewindableExact>> {
        let mut c: State<RewindableExact> = match ring {
            Some(k) => State::<RewindableExact>::in_ring(&cfg, &model.spec_pools(1, k, &dev))?,
            None => RewindableExact::single(&model, &dev, 8)?,
        };
        c.set_tap_layers(&taps)?;
        Ok(c)
    };
    let (mut compared, mut kept, mut moved) = (0usize, 0usize, 0usize);
    for cell in cells(5, 9) {
        for seed in [0u64, 7, 42] {
            let reference = k1(
                &model,
                RewindableExact::single(&model, &dev, 8)?,
                &dev,
                &eos,
                &cell,
                seed,
                &prompt,
                max,
            )?;
            moved += usize::from(reference.0 != greedy_ids);
            let mut runs: Vec<crate::sampled_cells::LabelledRun> = Vec::new();
            for k in [4usize, 2] {
                for ring in [None, Some(k)] {
                    let drafter = tiny_drafter(drafter_cfg(&cfg, 4), &dev, 3)?;
                    let mut d = DSparkSpeculator::with_head_ctx(
                        drafter,
                        QwenHead::of(&model),
                        1,
                        CtxKv::Contiguous,
                    )?;
                    d.set_max_draft(k);
                    let got = spec(
                        &model,
                        cache(ring)?,
                        &dev,
                        &eos,
                        &cell,
                        seed,
                        &prompt,
                        max,
                        &mut d,
                    )?;
                    runs.push((format!("dflash k={k} ring={ring:?}"), got));
                }
            }
            for (name, stream) in [("oracle", Some(reference.0.clone())), ("garbage", None)] {
                let mut s = Scripted {
                    stream,
                    k: 3,
                    produced: 0,
                    rows: 0,
                    seeded: false,
                };
                let got = spec(
                    &model,
                    cache(Some(3))?,
                    &dev,
                    &eos,
                    &cell,
                    seed,
                    &prompt,
                    max,
                    &mut s,
                )?;
                runs.push((name.into(), got));
            }
            for (name, got) in runs {
                assert_eq!(
                    (&got.0, got.1),
                    (&reference.0, reference.1),
                    "{dev:?} {cell:?} seed {seed}: the {name} run is not the K=1 run"
                );
                kept += got.2;
                compared += 1;
            }
        }
    }
    println!("DFLASH-SAMPLED {dev:?} runs={compared} kept_drafts={kept} sampled_moved={moved}");
    assert!(kept > compared / 2, "the drafts were rarely kept ({kept})");
    assert!(moved > 10, "the cells rarely left greedy's stream");
    Ok(())
}

#[test]
fn sampled_dflash_on_qwen_is_the_k1_run() -> Result<()> {
    sampled_identity_on(&Device::Cpu)
}

#[cfg(feature = "metal")]
#[test]
fn sampled_dflash_on_qwen_is_the_k1_run_on_metal() -> Result<()> {
    sampled_identity_on(&Device::new_metal(0)?)
}
