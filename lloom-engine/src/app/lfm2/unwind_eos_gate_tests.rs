//! A decode forward that unwinds with the EOS gate armed, on the actual
//! single-request owner: the gate's guard ends the gate during the unwind,
//! which waits for the queue and closes the open encoder there. That close
//! completes, so the next request on the device must succeed. Gate off is
//! the control. Run per API: `LLOOM_METAL_API=metal3|metal4`.
use super::*;
use lloom::models::lfm2_testing::Lfm2Tiny;
use lloom::nn::model::{CausalLM, Model as ModelTrait};
use lloom::testing::tiny::{TinyCheckpoint, TinyConfig, TinyFamily, Weights};
use std::sync::atomic::AtomicUsize;

type Owner = lloom::profile::CrownExact;
type Inner = Model<Owner>;

struct FixtureDir(PathBuf);
impl Drop for FixtureDir {
    fn drop(&mut self) {
        // Only this test's freshly created checkpoint directory is owned here.
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn fixture(eos_gate: bool) -> Result<(FixtureDir, Engine<Owner>)> {
    let requested = std::env::var("LLOOM_METAL_API")
        .context("LLOOM_METAL_API=metal3|metal4 selects the API this runs on")?;
    anyhow::ensure!(matches!(requested.as_str(), "metal3" | "metal4"));
    // Install the tuning before the checkpoint is written: the writer builds a
    // CPU schema model, which fixes the regime.
    let mut options = crate::EngineOptions::default();
    options.tuning.contract.profile = "deterministic".into();
    options.tuning.metal.command_api = requested.clone();
    options.tuning.install().map_err(anyhow::Error::msg)?;
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!(
        "lloom-unwind-eos-gate-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir)?;
    let owner = FixtureDir(dir.clone());
    let mut cfg = Lfm2Tiny::template();
    assert!(TinyConfig::set_axis(
        &mut cfg,
        "vocab_size",
        Lfm2Tiny::tokenizer().vocab_size()
    ));
    cfg.derive();
    let model_dir = TinyCheckpoint::<Lfm2Tiny>::new(cfg, Weights::default()).write(&dir)?;
    let mut decode = crate::DecodeConfig::default();
    decode.readback.eos_gate = eos_gate;
    let engine = Engine::<Owner>::load_as(LoadOptions {
        model_dir,
        tokenizer: None,
        cpu: false,
        purpose: lloom::nn::attention::route::ExecutionPurpose::Production,
        kv_capacity: 256,
        readback: ReadbackArm::Auto,
        decode,
        model: crate::ModelConfig {
            cache: "crown".into(),
            ..Default::default()
        },
        reuse_prefix: true,
        pin_prefix: false,
        warmup: false,
        seed_text: "Hello".into(),
        model_name: Some("tiny-unwind-eos-gate".into()),
        batch: 1,
        intake_per_tick: crate::options::DEFAULT_INTAKE_PER_TICK,
        dspark: None,
        prefill_tile: 16,
        prefill_chunk: 0,
        memory_fraction: 0.0,
        mem_sample: false,
        rows: RowsArm::Private,
        share_prefix: false,
        dspark_k: None,
        executor: Default::default(),
        memory: Default::default(),
        flash_k: None,
        decision_template: None,
        decision_ragged: false,
    })?;
    let actual = engine
        .core
        .device
        .as_metal_device()?
        .commands()
        .actual_command_api();
    assert_eq!(actual, requested, "the fixture runs on the API it names");
    println!("UNWIND_EOS_GATE actual_api={actual} eos_gate={eos_gate}");
    Ok((owner, engine))
}

fn params() -> DecodeParams {
    DecodeParams {
        max_tokens: 16,
        eos_ids: NEVER_EOS.to_vec(),
        readback: ReadbackArm::Ring(crate::RingArm::CROWN),
        ..Default::default()
    }
}

/// The loaded model, with a panic in every forward from position `from` on.
struct PanicFrom<'a> {
    inner: &'a Inner,
    from: usize,
    panicked: AtomicUsize,
}

impl PanicFrom<'_> {
    fn at(&self, pos: usize) {
        if pos >= self.from {
            self.panicked.fetch_add(1, Ordering::Relaxed);
            panic!("injected decode-forward panic at {pos}");
        }
    }
}

impl ModelTrait for PanicFrom<'_> {
    type Config = Config;
    fn from_weights(_: VarBuilder, _: &Config) -> lloom_tensor::Result<Self> {
        lloom_tensor::bail!("test wrapper borrows the loaded model")
    }
}

impl CausalLM for PanicFrom<'_> {
    type Cache = <Inner as CausalLM>::Cache;
    fn new_cache(&self, cfg: &Config, dev: &Device) -> lloom_tensor::Result<Self::Cache> {
        self.inner.new_cache(cfg, dev)
    }
    fn embed(&self, ids: &Tensor) -> lloom_tensor::Result<Tensor> {
        self.inner.embed(ids)
    }
    fn forward(
        &self,
        ids: &Tensor,
        pos: usize,
        cache: &mut Self::Cache,
    ) -> lloom_tensor::Result<Tensor> {
        let logits = self.inner.forward(ids, pos, cache)?;
        self.at(pos);
        Ok(logits)
    }
    fn forward_embeds(
        &self,
        x: &Tensor,
        pos: usize,
        cache: &mut Self::Cache,
    ) -> lloom_tensor::Result<Tensor> {
        let logits = self.inner.forward_embeds(x, pos, cache)?;
        self.at(pos);
        Ok(logits)
    }
    fn prepare_pass(&self, cache: &mut Self::Cache, n: usize) -> lloom_tensor::Result<()> {
        self.inner.prepare_pass(cache, n)
    }
    fn open_checkpoint_request(
        &self,
        cache: &mut Self::Cache,
        policy: lloom::nn::CheckpointPolicy,
        end: usize,
    ) -> lloom_tensor::Result<()> {
        self.inner.open_checkpoint_request(cache, policy, end)
    }
    fn close_checkpoint_request(&self, cache: &mut Self::Cache) -> lloom_tensor::Result<()> {
        self.inner.close_checkpoint_request(cache)
    }
    fn pin_checkpoint(
        &self,
        cache: &mut Self::Cache,
        pos: Option<usize>,
    ) -> lloom_tensor::Result<Option<usize>> {
        self.inner.pin_checkpoint(cache, pos)
    }
    fn pins_checkpoints(&self, cache: &Self::Cache) -> bool {
        self.inner.pins_checkpoints(cache)
    }
    fn pinned_checkpoint(&self, cache: &Self::Cache) -> Option<usize> {
        self.inner.pinned_checkpoint(cache)
    }
    fn eos_gate_cores(&self) -> bool {
        self.inner.eos_gate_cores()
    }
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
}

fn run(engine: &Engine<Owner>, prompt: &[u32]) -> Result<Decoded> {
    engine.complete_until(prompt, &params(), |_| true)
}

/// A decode forward panics past the prompt, inside the owner's lease; the
/// next request on the same engine decodes what an undisturbed one does.
fn a_decode_forward_panic_leaves_the_next_request_working(eos_gate: bool) -> Result<()> {
    let (_files, engine) = fixture(eos_gate)?;
    let prompt = engine.encode("Hello world", false)?;
    let reference = run(&engine, &prompt)?;
    assert_eq!(reference.tokens.len(), params().max_tokens);
    let fault = PanicFrom {
        inner: &engine.core.model,
        from: prompt.len() + 2,
        panicked: AtomicUsize::new(0),
    };
    let request = params();
    let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<_> {
        let mut lease = engine.checkout_single()?;
        let decode = lease.extra().decode.clone();
        Ok(
            lease.with_model(&fault, &engine.core.device, decode, |session, _| {
                session.decode_until(&prompt, engine.detok(), &request, |_| true)
            })?,
        )
    }));
    assert!(unwound.is_err(), "the injected forward did not unwind");
    assert_eq!(fault.panicked.load(Ordering::Relaxed), 1);
    let failure = engine
        .core
        .device
        .as_metal_device()?
        .commands()
        .check_failed();
    println!("UNWIND_EOS_GATE eos_gate={eos_gate} queue_after_unwind={failure:?}");
    let next = run(&engine, &prompt)
        .with_context(|| format!("the request after an unwind (eos_gate={eos_gate}) failed"))?;
    assert_eq!(next.tokens, reference.tokens, "eos_gate={eos_gate}");
    println!(
        "UNWIND_EOS_GATE eos_gate={eos_gate} next=ok tokens={}",
        next.tokens.len()
    );
    Ok(())
}

#[test]
fn a_decode_forward_panic_with_the_eos_gate_armed_leaves_the_next_request_working() -> Result<()> {
    a_decode_forward_panic_leaves_the_next_request_working(true)
}

#[test]
fn a_decode_forward_panic_without_the_eos_gate_leaves_the_next_request_working() -> Result<()> {
    a_decode_forward_panic_leaves_the_next_request_working(false)
}
