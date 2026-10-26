//! A decode step whose pass the device refuses runs no forward
//! (`decode.rs`'s `begin_decode_step`, `DESIGN.md` §9.2aa).
//!
//! `MetalDevice::begin_pass` is fallible: it declines a pass whose
//! predecessor was never closed and one whose cached plan it cannot build.
//! A loop that discards that refusal encodes its forward anyway — against
//! the allocation state the pass was meant to install, with the pool's
//! arena cleared and the refused pass's bookkeeping absent.
//!
//! The refusal injected here is the device's own already-open-pass arm,
//! reached through the public API: the model's `prepare_pass` — which the
//! loop calls immediately before it opens the step's pass — opens a pass
//! of its own at a chosen step and leaves it open. The witness is the
//! wrapper's forward count: the refused step must not forward. The
//! injection's own stray pass is closed by the harness afterwards, and the
//! next request over the same sequence must decode exactly what a run
//! without the injection decodes.
#![cfg(feature = "metal")]

use lloom::generation::{Blocking, LogitsProcessor, Sampling};
use lloom::models::lfm2::{Config, ConvState, KvAppend, LayerType, Model, State};
use lloom::models::lfm2_testing::Lfm2Tiny;
use lloom::nn::model::{CausalLM, Model as ModelTrait, Provisional};
use lloom::nn::{DecodeOptions, IdentityProposer, Session, VarBuilder};
use lloom::profile::RewindableExact;
use lloom::testing::tiny::{TinyCheckpoint, Weights};
use lloom_tensor::metal_backend::{ArenaLayout, PassKey, PassKind};
use lloom_tensor::{DType, Device, Result, Tensor};
use std::sync::atomic::{AtomicUsize, Ordering};

type Inner = Model<RewindableExact>;

const EOS: u32 = 47;
const PROMPT: [u32; 4] = [1, 2, 3, 4];

fn config() -> Config {
    Config {
        vocab_size: 48,
        hidden_size: 128,
        intermediate_size: 256,
        num_hidden_layers: 3,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        norm_eps: 1e-5,
        rope_theta: 10000.0,
        max_position_embeddings: 128,
        conv_l_cache: 3,
        conv_bias: false,
        layer_types: vec![LayerType::Conv, LayerType::FullAttention, LayerType::Conv],
        tie_embedding: true,
        bos_token_id: None,
        eos_token_id: Some(EOS),
        kv_append: KvAppend::InPlace,
        conv_state: ConvState::RotatingRing { k: 2 },
        conv_prefill_ctx: true,
        ..Default::default()
    }
}

/// The tiny LFM2 with an injection in `prepare_pass`, the call the decode
/// loop makes immediately before it opens the step's pass: armed with `n`,
/// the `n`-th single-position `prepare_pass` from the arming opens a
/// `Decode { b: 1 }` pass on the device and leaves it open, so the loop's
/// own `begin_pass` meets a pass that is already open. Every forward is
/// counted.
struct StrayPass {
    inner: Inner,
    device: Device,
    left: AtomicUsize,
    opened: AtomicUsize,
    forwards: AtomicUsize,
}

impl StrayPass {
    fn arm(&self, n: Option<usize>) {
        self.left.store(n.unwrap_or(0), Ordering::Relaxed);
    }

    fn forwards(&self) -> usize {
        self.forwards.load(Ordering::Relaxed)
    }

    fn opened(&self) -> usize {
        self.opened.load(Ordering::Relaxed)
    }

    /// Close a stray pass the injection left open.
    fn close_stray(&self) -> Result<()> {
        let Device::Metal(m) = &self.device else {
            unreachable!("a Metal fixture")
        };
        if m.pass_open() {
            m.finish_pass()?;
        }
        Ok(())
    }
}

impl ModelTrait for StrayPass {
    type Config = Config;
    fn from_weights(_vb: VarBuilder, _cfg: &Config) -> Result<Self> {
        lloom_tensor::bail!("built from the tiny checkpoint")
    }
}

impl CausalLM for StrayPass {
    type Cache = <Inner as CausalLM>::Cache;
    fn new_cache(&self, cfg: &Config, dev: &Device) -> Result<Self::Cache> {
        self.inner.new_cache(cfg, dev)
    }
    fn embed(&self, tokens: &Tensor) -> Result<Tensor> {
        self.inner.embed(tokens)
    }
    fn forward_embeds(
        &self,
        embeds: &Tensor,
        pos: usize,
        cache: &mut Self::Cache,
    ) -> Result<Tensor> {
        self.forwards.fetch_add(1, Ordering::Relaxed);
        self.inner.forward_embeds(embeds, pos, cache)
    }
    fn prepare_pass(&self, cache: &mut Self::Cache, seq_len: usize) -> Result<()> {
        self.inner.prepare_pass(cache, seq_len)?;
        if seq_len != 1 {
            return Ok(());
        }
        let left = self.left.load(Ordering::Relaxed);
        if left == 0 {
            return Ok(());
        }
        self.left.store(left - 1, Ordering::Relaxed);
        if left == 1 {
            let Device::Metal(m) = &self.device else {
                unreachable!("a Metal fixture")
            };
            m.begin_pass(PassKey::new(PassKind::Decode { b: 1 }))?;
            self.opened.fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }
    fn note_input_ids(&self, cache: &mut Self::Cache, ids: &[u32]) -> Result<()> {
        self.inner.note_input_ids(cache, ids)
    }
    fn checkpoint_at(&self, cache: &mut Self::Cache, pos: usize) -> Result<()> {
        self.inner.checkpoint_at(cache, pos)
    }
    fn settle_pending(&self, cache: &mut Self::Cache) -> Result<()> {
        self.inner.settle_pending(cache)
    }
    fn provisional(&self, cache: &mut Self::Cache, mode: Provisional) -> Result<Provisional> {
        self.inner.provisional(cache, mode)
    }
    fn settle_step(&self, cache: &mut Self::Cache) -> Result<Option<Tensor>> {
        self.inner.settle_step(cache)
    }
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
}

fn model(dev: &Device) -> Result<(StrayPass, Config)> {
    let cfg = config();
    let inner = TinyCheckpoint::<Lfm2Tiny>::new(cfg.clone(), Weights::default())
        .with_dtype(DType::F16)
        .model::<RewindableExact>(dev)?;
    Ok((
        StrayPass {
            inner,
            device: dev.clone(),
            left: AtomicUsize::new(0),
            opened: AtomicUsize::new(0),
            forwards: AtomicUsize::new(0),
        },
        cfg,
    ))
}

/// One greedy request over its own sequence, through the blocking readback.
fn request(model: &StrayPass, cfg: &Config, dev: &Device, max: usize) -> Result<Vec<u32>> {
    let mut session = Session::new(
        model,
        State::<RewindableExact>::new(true, DType::F16, cfg, dev)?,
    );
    let mut readback = Blocking::new(&[EOS], dev)?;
    let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
    let out = session.decode(
        &mut IdentityProposer,
        &mut readback,
        &mut sampler,
        &PROMPT,
        &DecodeOptions::new(max),
        &mut |_| Ok(()),
    )?;
    Ok(out.ids)
}

/// The decode step whose pass the device refuses encodes no forward, says
/// why, and leaves the device with nothing of its own open — so the next
/// request decodes what an uninjected run decodes.
#[test]
fn a_refused_decode_pass_runs_no_forward_and_the_next_request_is_usable() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let Device::Metal(m) = &dev else {
        unreachable!()
    };
    let (model, cfg) = model(&dev)?;
    m.enable_pass_plans(ArenaLayout::Packed);

    // The reference: the same request with nothing injected.
    let reference = request(&model, &cfg, &dev, 6)?;
    assert!(!reference.is_empty(), "the fixture decodes something");

    // The third decode step's pass is refused: `prepare_pass` opens one
    // first and leaves it open.
    let before = model.forwards();
    model.arm(Some(3));
    let refused = request(&model, &cfg, &dev, 6);
    let during = model.forwards();
    model.arm(None);
    assert_eq!(model.opened(), 1, "the injection opened its stray pass");
    let why = refused
        .expect_err("a refused pass fails the request")
        .to_string();
    assert!(
        why.contains("a pass is already open"),
        "the refusal is the device's, named: {why}"
    );
    // The prompt's prefill forward plus the two decode steps before the
    // refused one: the refused step itself encodes nothing.
    assert_eq!(
        during - before,
        3,
        "the step whose pass was refused forwarded anyway"
    );

    // The injection's own stray pass is the harness's to close; the loop
    // opened nothing on top of it.
    model.close_stray()?;
    assert!(!m.pass_open(), "the refused step left a pass open");

    let after = request(&model, &cfg, &dev, 6)?;
    assert_eq!(
        after, reference,
        "the request after a refused pass decoded something else"
    );
    Ok(())
}
