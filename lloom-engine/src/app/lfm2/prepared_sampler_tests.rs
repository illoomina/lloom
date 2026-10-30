// Native prepared sampling through the real single-request owner.
// Candidate only: compile and native execution are separate gates.
use super::*;
use lloom::models::lfm2_testing::Lfm2Tiny;
use lloom::nn::model::{CausalLM, Model as ModelTrait};
use lloom::profile::{Profile, Regime};
use lloom::testing::tiny::{TinyCheckpoint, TinyConfig, TinyFamily, Weights};
use std::sync::atomic::{AtomicUsize, Ordering};

type NativeEngine = Engine<OwnerProfile>;
type Inner = Model<OwnerProfile>;

struct FixtureDir(PathBuf);
impl Drop for FixtureDir {
    fn drop(&mut self) {
        // Only this test's freshly created checkpoint directory is owned here.
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn fixture(decode: crate::DecodeConfig) -> Result<(FixtureDir, NativeEngine)> {
    fixture_with_trace(decode, false)
}

fn fixture_with_trace(
    decode: crate::DecodeConfig,
    trace: bool,
) -> Result<(FixtureDir, NativeEngine)> {
    let requested = std::env::var("LLOOM_METAL_API")?;
    anyhow::ensure!(
        matches!(requested.as_str(), "metal3" | "metal4"),
        "explicit Metal API required"
    );
    let mut options = crate::EngineOptions::default();
    let expected_contract = <<OwnerProfile as Profile>::Regime as Regime>::CONTRACT;
    let contract = crate::decode::ring_capture::fixture_contract(Some(expected_contract.name()))?;
    assert_eq!(contract, expected_contract);
    options.tuning.contract.profile = contract.name().into();
    options.tuning.metal.command_api = requested.clone();
    options.tuning.instrument.profile = true;
    options.tuning.instrument.trace = trace;
    options.tuning.install().map_err(anyhow::Error::msg)?;
    // Read cached knobs only after installing the complete fixture config.
    assert!(lloom_tensor::metal_backend::profile::enabled());
    let actual_contract = lloom_tensor::tuning::contract();
    assert_eq!(actual_contract, expected_contract);
    println!(
        "PREPARED_OWNER_CONFIG actual_contract={} tensor_tuning={}",
        actual_contract.name(),
        serde_json::to_string(lloom_tensor::tuning::tuning())?
    );
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!(
        "lloom-prepared-owner-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir(&dir)?;
    let owner = FixtureDir(dir.clone());
    let mut cfg = Lfm2Tiny::template();
    assert!(TinyConfig::set_axis(
        &mut cfg,
        "vocab_size",
        Lfm2Tiny::tokenizer().vocab_size()
    ));
    cfg.derive();
    // Preserve TinyCheckpoint::write's directory preparation before the typed writer.
    std::fs::create_dir_all(&dir)?;
    Lfm2Tiny::write_for::<OwnerProfile>(
        &TinyCheckpoint::<Lfm2Tiny>::new(cfg, Weights::default()),
        &dir,
    )?;
    let opts = LoadOptions {
        model_dir: dir,
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
        model_name: Some("tiny-prepared-owner".into()),
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
        text_only: true,
    };
    let engine = NativeEngine::load_as(opts)?;
    assert!(
        engine.core.device.is_metal(),
        "native test cannot fall back to CPU"
    );
    assert!(engine.sched.is_none());
    assert_eq!(
        engine
            .core
            .device
            .as_metal_device()?
            .commands()
            .actual_command_api(),
        requested
    );
    println!(
        "PREPARED_OWNER actual_api={:?} actual_contract={}",
        engine
            .core
            .device
            .as_metal_device()?
            .commands()
            .actual_command_api(),
        lloom_tensor::tuning::contract().name()
    );
    Ok((owner, engine))
}

fn params(arm: ReadbackArm, gumbel: bool, cold: bool) -> DecodeParams {
    DecodeParams {
        seed: 781,
        temperature: Some(0.7),
        top_k: Some(8),
        top_p: Some(0.8),
        gumbel,
        max_tokens: 32,
        eos_ids: NEVER_EOS.to_vec(),
        readback: arm,
        cold,
        ..Default::default()
    }
}

fn run(engine: &NativeEngine, ids: &[u32], params: &DecodeParams) -> Result<Decoded> {
    let mut text = String::new();
    let out = engine.complete_until(ids, params, |part| {
        text.push_str(part);
        true
    })?;
    assert_eq!(text, out.text);
    assert_eq!(out.tokens.len(), params.max_tokens);
    assert_eq!(out.stop, Stop::Length);
    Ok(out)
}

#[test]
#[ignore = "explicit native admission; CANDLE_METAL_PROFILE=1; run per Metal API"]
fn prepared_sampler_actual_owner_matches_blocking_across_cadences() -> Result<()> {
    let mut decode = crate::DecodeConfig::default();
    decode.readback.device_filter = true;
    decode.readback.run_ahead = 3;
    decode.readback.queued_steps = 2;
    let (_files, engine) = fixture(decode)?;
    let prompt = engine.encode("Hello world", false)?;
    assert!(prompt.len() > 1);
    for gumbel in [false, true] {
        let reference = run(
            &engine,
            &prompt,
            &params(ReadbackArm::Blocking, gumbel, true),
        )?;
        for arm in [
            crate::RingArm {
                lookahead: 0,
                adaptive: false,
                ..crate::RingArm::FIXED
            },
            crate::RingArm::FIXED,
            crate::RingArm::CROWN,
            crate::RingArm {
                eos: lloom::generation::EosRead::HostOnDrain,
                ..crate::RingArm::FIXED
            },
            crate::RingArm {
                eos: lloom::generation::EosRead::GpuBatched,
                ..crate::RingArm::FIXED
            },
            crate::RingArm {
                eos: lloom::generation::EosRead::GpuBatchedNonblocking,
                ..crate::RingArm::FIXED
            },
            crate::RingArm {
                eos: lloom::generation::EosRead::HostOnDrain,
                ..crate::RingArm::CROWN
            },
            crate::RingArm {
                eos: lloom::generation::EosRead::GpuBatched,
                ..crate::RingArm::CROWN
            },
            crate::RingArm {
                eos: lloom::generation::EosRead::GpuBatchedNonblocking,
                ..crate::RingArm::CROWN
            },
        ] {
            let census = engine
                .core
                .device
                .as_metal_device()?
                .dispatch_census()
                .open()
                .expect("one isolated route census");
            let out = run(
                &engine,
                &prompt,
                &params(ReadbackArm::Ring(arm), gumbel, true),
            )?;
            engine.core.device.synchronize()?;
            let labels = census.close();
            assert!(
                labels
                    .iter()
                    .any(|(name, count)| name.contains("sampling_topk_mask") && *count > 0),
                "prepared sampler did not dispatch: {labels:?}"
            );
            assert!(matches!(out.readback, ReadbackArm::Ring(_)));
            assert!(out
                .readback_note
                .as_deref()
                .is_some_and(|s| s.contains("native prepared")));
            assert_eq!(out.tokens, reference.tokens, "gumbel={gumbel} arm={arm:?}");
            assert_eq!(out.text, reference.text);
            println!("PREPARED_OWNER gumbel={gumbel} requested={arm:?} actual={:?} tokens={:?} labels={labels:?}", out.readback, out.tokens);
        }
    }
    Ok(())
}

struct InvalidLogits<'a> {
    inner: &'a Inner,
    from: usize,
    panic: bool,
    injected: AtomicUsize,
    prepared: AtomicUsize,
    observer: Option<&'a BodyObserver>,
}
impl ModelTrait for InvalidLogits<'_> {
    type Config = Config;
    fn from_weights(_: VarBuilder, _: &Config) -> lloom_tensor::Result<Self> {
        lloom_tensor::bail!("test wrapper borrows the loaded model")
    }
}
impl InvalidLogits<'_> {
    fn observe(
        &self,
        route: &'static str,
        input: &Tensor,
        pos: usize,
        body: impl FnOnce() -> lloom_tensor::Result<Tensor>,
    ) -> lloom_tensor::Result<Tensor> {
        match self.observer {
            Some(observer) => observer.observe(route, input, pos, body),
            None => body(),
        }
    }
    fn corrupt(&self, logits: Tensor, pos: usize) -> lloom_tensor::Result<Tensor> {
        if pos >= self.from {
            self.injected.fetch_add(1, Ordering::Relaxed);
            assert!(!self.panic, "injected prepared producer panic");
            Tensor::full(f32::NAN, logits.shape(), logits.device())?.to_dtype(logits.dtype())
        } else {
            Ok(logits)
        }
    }
}
impl CausalLM for InvalidLogits<'_> {
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
        self.corrupt(
            self.observe("token", ids, pos, || self.inner.forward(ids, pos, cache))?,
            pos,
        )
    }
    fn forward_embeds(
        &self,
        x: &Tensor,
        pos: usize,
        cache: &mut Self::Cache,
    ) -> lloom_tensor::Result<Tensor> {
        self.corrupt(
            self.observe("embeds", x, pos, || {
                self.inner.forward_embeds(x, pos, cache)
            })?,
            pos,
        )
    }
    fn guarded_embedding_layout(
        &self,
    ) -> lloom_tensor::Result<lloom_tensor::quantized::GuardedEmbeddingLayout> {
        self.inner.guarded_embedding_layout()
    }
    fn check_guarded_embedding(&self, out: &Tensor) -> lloom_tensor::Result<()> {
        self.inner.check_guarded_embedding(out)
    }
    fn embed_guarded_into(
        &self,
        ids: &lloom_tensor::metal_backend::sampling::GuardedSample<'_>,
        out: &Tensor,
    ) -> lloom_tensor::Result<()> {
        self.inner.embed_guarded_into(ids, out)
    }
    fn forward_prepared_embeds(
        &self,
        x: &Tensor,
        pos: usize,
        cache: &mut Self::Cache,
    ) -> lloom_tensor::Result<Tensor> {
        self.prepared.fetch_add(1, Ordering::Relaxed);
        self.corrupt(
            self.observe("prepared", x, pos, || {
                self.inner.forward_prepared_embeds(x, pos, cache)
            })?,
            pos,
        )
    }
    fn prepare_pass(&self, cache: &mut Self::Cache, n: usize) -> lloom_tensor::Result<()> {
        self.inner.prepare_pass(cache, n)
    }
    fn checkpoint_at(&self, cache: &mut Self::Cache, pos: usize) -> lloom_tensor::Result<()> {
        self.inner.checkpoint_at(cache, pos)
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
    fn eos_gate_cores(&self) -> bool {
        self.inner.eos_gate_cores()
    }
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
}

#[test]
#[ignore = "explicit native admission; run per Metal API"]
fn invalid_prepared_row_recovers_through_actual_lfm_request_owner() -> Result<()> {
    let mut decode = crate::DecodeConfig::default();
    decode.readback.device_filter = true;
    decode.readback.run_ahead = 3;
    decode.readback.queued_steps = 2;
    let (_files, engine) = fixture(decode)?;
    let prompt = engine.encode("Hello world", false)?;
    let fault = InvalidLogits {
        inner: &engine.core.model,
        from: prompt.len() + 2,
        panic: false,
        injected: AtomicUsize::new(0),
        prepared: AtomicUsize::new(0),
        observer: None,
    };
    let request = params(ReadbackArm::Ring(crate::RingArm::CROWN), false, false);
    let err = {
        let mut lease = engine.checkout_single()?;
        let decode = lease.extra().decode.clone();
        let result = lease.with_model(&fault, &engine.core.device, decode, |session, _| {
            session.decode_until(&prompt, engine.detok(), &request, |_| true)
        });
        result.expect_err("invalid device row must fail the request")
    };
    assert!(fault.injected.load(Ordering::Relaxed) > 0);
    assert!(
        fault.prepared.load(Ordering::Relaxed) > 0,
        "fault bypassed prepared route"
    );
    assert!(
        err.to_string().contains("sampling row 0: Nan"),
        "wrong failure: {err}"
    );
    println!("PREPARED_OWNER invalid_error={err}");
    assert!(engine.single.lock().as_ref().unwrap().needs_recovery);
    // No test-side reset or fence: the real owner must fence/reset on checkout.
    let next = run(&engine, &prompt, &request)?;
    assert!(!engine.single.lock().as_ref().unwrap().needs_recovery);
    assert_eq!(next.reused, 0);
    let reference = run(
        &engine,
        &prompt,
        &params(ReadbackArm::Blocking, false, true),
    )?;
    assert_eq!(next.tokens, reference.tokens);
    assert_eq!(next.text, reference.text);
    println!("PREPARED_OWNER recovery=pass tokens={:?}", next.tokens);
    Ok(())
}

/// The owner population: an invalid prepared row, the owner's cleanup, then
/// the next public requests (cold, then reusing the prefix) against the
/// checked Blocking comparator's own cold and reused-prefix pair. The
/// comparator runs before the fault or after the recovery, so each input
/// route records its decode plan first in one of the two orders.
fn invalid_row_recovery_population(comparator_first: bool) -> Result<()> {
    let mut decode = crate::DecodeConfig::default();
    decode.readback.device_filter = true;
    decode.readback.run_ahead = 3;
    decode.readback.queued_steps = 2;
    let (_files, engine) = fixture(decode)?;
    let prompt = engine.encode("Hello world", false)?;
    let ring = params(ReadbackArm::Ring(crate::RingArm::CROWN), false, false);
    let blocking = params(ReadbackArm::Blocking, false, false);
    let pair = |request: &DecodeParams| -> Result<(Decoded, Decoded)> {
        let cold = run(
            &engine,
            &prompt,
            &DecodeParams {
                cold: true,
                ..request.clone()
            },
        )?;
        let warm = run(&engine, &prompt, request)?;
        Ok((cold, warm))
    };
    let before = if comparator_first {
        Some(pair(&blocking)?)
    } else {
        None
    };
    let fault = InvalidLogits {
        inner: &engine.core.model,
        from: prompt.len() + 2,
        panic: false,
        injected: AtomicUsize::new(0),
        prepared: AtomicUsize::new(0),
        observer: None,
    };
    let err = {
        let mut lease = engine.checkout_single()?;
        let decode = lease.extra().decode.clone();
        let result = lease.with_model(&fault, &engine.core.device, decode, |session, _| {
            session.decode_until(&prompt, engine.detok(), &ring, |_| true)
        });
        result.expect_err("invalid device row must fail the request")
    };
    assert!(fault.injected.load(Ordering::Relaxed) > 0);
    assert!(
        fault.prepared.load(Ordering::Relaxed) > 0,
        "fault bypassed prepared route"
    );
    assert!(
        err.to_string().contains("sampling row 0: Nan"),
        "wrong failure: {err}"
    );
    assert!(engine.single.lock().as_ref().unwrap().needs_recovery);
    let cold = run(&engine, &prompt, &ring)?;
    assert!(!engine.single.lock().as_ref().unwrap().needs_recovery);
    assert_eq!(cold.reused, 0, "owner recovery must reset the sequence");
    let warm = run(&engine, &prompt, &ring)?;
    assert!(warm.reused > 0, "second request did not reuse the prefix");
    let (reference_cold, reference_warm) = match before {
        Some(pair) => pair,
        None => pair(&blocking)?,
    };
    assert_eq!(reference_cold.reused, 0);
    assert_eq!(reference_warm.reused, warm.reused);
    for out in [&cold, &warm] {
        assert!(matches!(out.readback, ReadbackArm::Ring(_)));
    }
    for out in [&reference_cold, &reference_warm] {
        assert_eq!(out.readback, ReadbackArm::Blocking);
        assert!(out
            .readback_note
            .as_deref()
            .is_some_and(|note| note.contains("checked blocking id")));
    }
    assert_eq!(cold.tokens, reference_cold.tokens, "cold after recovery");
    assert_eq!(cold.text, reference_cold.text);
    assert_eq!(
        warm.tokens, reference_warm.tokens,
        "reused prefix after recovery"
    );
    assert_eq!(warm.text, reference_warm.text);
    println!(
        "PREPARED_OWNER_POPULATION comparator_first={comparator_first} reused={} cold={:?} warm={:?}",
        warm.reused, cold.tokens, warm.tokens
    );
    Ok(())
}

#[test]
#[ignore = "explicit native admission; run per Metal API"]
fn invalid_row_recovery_against_a_later_blocking_comparator() -> Result<()> {
    invalid_row_recovery_population(false)
}

#[test]
#[ignore = "explicit native admission; run per Metal API"]
fn invalid_row_recovery_against_an_earlier_blocking_comparator() -> Result<()> {
    invalid_row_recovery_population(true)
}

#[test]
#[ignore = "explicit native admission; unwind build; external timeout; run per Metal API"]
fn prepared_producer_unwind_recovers_through_actual_owner() -> Result<()> {
    let mut decode = crate::DecodeConfig::default();
    decode.readback.device_filter = true;
    decode.readback.run_ahead = 3;
    decode.readback.queued_steps = 2;
    let (_files, engine) = fixture(decode)?;
    let prompt = engine.encode("Hello world", false)?;
    let fault = InvalidLogits {
        inner: &engine.core.model,
        from: prompt.len() + 2,
        panic: true,
        injected: AtomicUsize::new(0),
        prepared: AtomicUsize::new(0),
        observer: None,
    };
    let request = params(ReadbackArm::Ring(crate::RingArm::CROWN), false, false);
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<_> {
        let mut lease = engine.checkout_single()?;
        let decode = lease.extra().decode.clone();
        Ok(
            lease.with_model(&fault, &engine.core.device, decode, |session, _| {
                session.decode_until(&prompt, engine.detok(), &request, |_| true)
            })?,
        )
    }));
    assert!(panicked.is_err(), "producer fault did not unwind");
    assert!(fault.injected.load(Ordering::Relaxed) > 0);
    assert!(fault.prepared.load(Ordering::Relaxed) > 0);
    assert!(engine.single.lock().as_ref().unwrap().needs_recovery);
    let next = run(&engine, &prompt, &request)?;
    assert_eq!(next.reused, 0);
    assert!(!engine.single.lock().as_ref().unwrap().needs_recovery);
    let reference = run(
        &engine,
        &prompt,
        &params(ReadbackArm::Blocking, false, true),
    )?;
    assert_eq!(next.tokens, reference.tokens);
    assert_eq!(next.text, reference.text);
    println!(
        "PREPARED_OWNER producer_unwind_recovery=pass tokens={:?}",
        next.tokens
    );
    Ok(())
}

#[test]
#[ignore = "explicit native admission; CANDLE_METAL_PROFILE=1; run per Metal API"]
fn prepared_unfiltered_owner_uses_native_blocking_including_diagnostics() -> Result<()> {
    let mut decode = crate::DecodeConfig::default();
    decode.readback.device_filter = true;
    let (_files, engine) = fixture(decode)?;
    let prompt = engine.encode("Hello world", false)?;
    for gumbel in [false, true] {
        let base = DecodeParams {
            seed: 781,
            temperature: Some(0.7),
            gumbel,
            max_tokens: 32,
            eos_ids: NEVER_EOS.to_vec(),
            cold: true,
            ..Default::default()
        };
        assert!(matches!(
            base.sampling()?,
            lloom::generation::Sampling::All { .. }
                | lloom::generation::Sampling::GumbelSoftmax { .. }
        ));
        let reference = run(
            &engine,
            &prompt,
            &DecodeParams {
                readback: ReadbackArm::CROWN,
                ..base.clone()
            },
        )?;
        for (digest, logprobs) in [(false, false), (true, false), (false, true), (true, true)] {
            let census = engine
                .core
                .device
                .as_metal_device()?
                .dispatch_census()
                .open()
                .expect("isolated native route census");
            let request = DecodeParams {
                readback: ReadbackArm::Blocking,
                logits_digest: digest,
                logprobs: logprobs.then(|| crate::LogprobRequest { top: 3, rows: true }),
                ..base.clone()
            };
            let out = run(&engine, &prompt, &request)?;
            engine.core.device.synchronize()?;
            let labels = census.close();
            let method = if gumbel {
                "sampling_filtered_gumbel"
            } else {
                "sampling_multinomial"
            };
            assert!(
                labels
                    .iter()
                    .any(|(name, count)| name.contains(method) && *count > 0),
                "native {method} missing: {labels:?}"
            );
            assert!(
                !labels
                    .iter()
                    .any(|(name, count)| name.contains("sample_inverse_cdf") && *count > 0),
                "incumbent All bypass: {labels:?}"
            );
            assert_eq!(out.readback, ReadbackArm::Blocking);
            assert!(out
                .readback_note
                .as_deref()
                .is_some_and(|note| note.contains("checked blocking id")));
            assert_eq!(
                out.tokens, reference.tokens,
                "no filters gumbel={gumbel} digest={digest} logprobs={logprobs}"
            );
            assert_eq!(out.text, reference.text);
            assert_eq!(out.logits_sha256.is_some(), digest);
            assert_eq!(out.logprobs.is_some(), logprobs);
            if let Some(capture) = out.logprobs {
                assert_eq!(capture.sampled.len(), out.tokens.len());
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "explicit native admission; run per Metal API"]
fn invalid_native_blocking_diagnostic_row_recovers_through_actual_owner() -> Result<()> {
    let mut decode = crate::DecodeConfig::default();
    decode.readback.device_filter = true;
    let (_files, engine) = fixture(decode)?;
    let prompt = engine.encode("Hello world", false)?;
    let fault = InvalidLogits {
        inner: &engine.core.model,
        from: prompt.len() + 2,
        panic: false,
        injected: AtomicUsize::new(0),
        prepared: AtomicUsize::new(0),
        observer: None,
    };
    let request = DecodeParams {
        seed: 781,
        temperature: Some(0.7),
        readback: ReadbackArm::Blocking,
        logits_digest: true,
        logprobs: Some(crate::LogprobRequest { top: 3, rows: true }),
        max_tokens: 32,
        eos_ids: NEVER_EOS.to_vec(),
        cold: false,
        ..Default::default()
    };
    let err = {
        let mut lease = engine.checkout_single()?;
        let decode = lease.extra().decode.clone();
        lease
            .with_model(&fault, &engine.core.device, decode, |session, _| {
                session.decode_until(&prompt, engine.detok(), &request, |_| true)
            })
            .expect_err("invalid native Blocking row must fail whole request")
    };
    assert!(
        fault.injected.load(Ordering::Relaxed) > 0,
        "actual post-forward failure reached"
    );
    assert_eq!(
        fault.prepared.load(Ordering::Relaxed),
        0,
        "Blocking retains ordinary token forward, not Ring prepared embedding"
    );
    assert!(
        err.to_string().contains("sampling row 0: Nan"),
        "native checked failure expected, not host fallback: {err}"
    );
    assert!(engine.single.lock().as_ref().unwrap().needs_recovery);
    // No fixture reset/fence: the actual next-request owner performs recovery.
    let next = run(&engine, &prompt, &request)?;
    assert!(!engine.single.lock().as_ref().unwrap().needs_recovery);
    assert_eq!(next.reused, 0);
    let reference = run(
        &engine,
        &prompt,
        &DecodeParams {
            cold: true,
            ..request.clone()
        },
    )?;
    assert_eq!(next.tokens, reference.tokens);
    assert_eq!(next.text, reference.text);
    assert_eq!(next.logits_sha256, reference.logits_sha256);
    assert_eq!(next.logprobs, reference.logprobs);
    Ok(())
}

#[test]
#[ignore = "explicit native admission; run per Metal API"]
fn prepared_owner_applies_and_clears_request_thinking_budget() -> Result<()> {
    let mut decode = crate::DecodeConfig::default();
    decode.readback.device_filter = true;
    decode.readback.run_ahead = 3;
    decode.readback.queued_steps = 2;
    let (_files, engine) = fixture(decode)?;
    assert!(engine.core.model.vocab_size() > 99);
    let prompt = engine.encode("Hello world", false)?;
    let budget = lloom::generation::budget::ThinkingBudget::new(0, vec![99], vec![98, 99])?;
    for gumbel in [false, true] {
        let base = DecodeParams {
            seed: 781,
            temperature: Some(0.7),
            top_k: Some(1),
            gumbel,
            processors: lloom::generation::Processors {
                logit_bias: vec![(97, 100.)],
                ..Default::default()
            },
            max_tokens: 8,
            eos_ids: NEVER_EOS.to_vec(),
            cold: true,
            ..Default::default()
        };
        let free = run(
            &engine,
            &prompt,
            &DecodeParams {
                readback: ReadbackArm::Blocking,
                ..base.clone()
            },
        )?;
        assert_eq!(
            free.tokens, [97; 8],
            "positive control: unforced draw is not either forced id"
        );
        let request = DecodeParams {
            thinking: Some(budget.clone()),
            readback: ReadbackArm::Blocking,
            ..base.clone()
        };
        let reference = run(&engine, &prompt, &request)?;
        assert_eq!(reference.tokens, [98, 99, 97, 97, 97, 97, 97, 97]);
        for arm in [
            ReadbackArm::RING_FIXED,
            ReadbackArm::Ring(crate::RingArm {
                lookahead: 0,
                commit_steps: 1,
                ..crate::RingArm::FIXED
            }),
            ReadbackArm::CROWN,
        ] {
            let out = run(
                &engine,
                &prompt,
                &DecodeParams {
                    readback: arm,
                    ..request.clone()
                },
            )?;
            assert_eq!(
                out.tokens, reference.tokens,
                "native ring omitted request thinking budget: {arm:?}"
            );
            assert_eq!(out.thinking, reference.thinking);
            let next = run(
                &engine,
                &prompt,
                &DecodeParams {
                    readback: arm,
                    ..base.clone()
                },
            )?;
            assert_eq!(
                next.tokens, free.tokens,
                "previous request's budget leaked: {arm:?}"
            );
            assert!(next.thinking.is_none());
        }
    }
    Ok(())
}

// The recorder is correctness instrumentation only: its device copies and
// host bookkeeping are absent from the shipping/price binary.
fn captured_run(
    engine: &NativeEngine,
    prompt: &[u32],
    request: &DecodeParams,
    capture: &crate::decode::ring_capture::Capture,
) -> Result<(Decoded, crate::decode::ring_capture::Completed)> {
    let mut text = String::new();
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        engine.with_single(|session, extra| {
            extra.image_keys.clear();
            session.reset();
            session.capture_ring_rows(capture.clone());
            session
                .decode_until(prompt, engine.detok(), request, |part| {
                    text.push_str(part);
                    true
                })
                .map_err(anyhow::Error::from)
        })
    }));
    // The fixture retains its own handle across request return AND unwind.
    // Complete outstanding copies before assertions or propagating failure.
    let terminal = capture.finish();
    let result = match outcome {
        Ok(result) => result,
        Err(payload) => {
            if let Err(error) = terminal {
                eprintln!("RING_CAPTURE_CLEANUP_FAILED {error}");
            }
            std::panic::resume_unwind(payload)
        }
    };
    let completed = terminal?;
    let out = result?;
    completed.check_matrix(
        prompt.len(),
        request.max_tokens,
        engine.core.model.vocab_size(),
    )?;
    assert_eq!(out.stats.steps, u64::try_from(request.max_tokens)?);
    assert_eq!(
        out.stats.redraws, 0,
        "ordinal map requires no provisional redraw"
    );
    assert_eq!(out.tokens.len(), request.max_tokens);
    assert_eq!(out.stop, Stop::Length);
    assert_eq!(out.text, text);
    assert!(matches!(out.readback, ReadbackArm::Ring(_)));
    assert!(out
        .readback_note
        .as_deref()
        .is_some_and(|s| s.contains("native prepared")));
    Ok((out, completed))
}

fn replay_captured_on_cpu(
    prompt: &[u32],
    request: &DecodeParams,
    completed: &crate::decode::ring_capture::Completed,
) -> Result<Vec<u32>> {
    use lloom::generation::{Blocking, ReadbackStrategy};
    let mut sampler = request.sampler()?;
    sampler.observe_prompt(prompt);
    let mut blocking = Blocking::new(&request.eos_ids, &Device::Cpu)?;
    blocking.set_thinking_budget(request.thinking.as_ref())?;
    let mut expected = Vec::with_capacity(request.max_tokens);
    let mut counts = vec![0usize; completed.rows[0].len()];
    for (ordinal, row) in completed.rows.iter().enumerate() {
        // Independent expected position; never trust the producer's receipt
        // as the oracle's noise key.
        let position = prompt
            .len()
            .checked_add(ordinal)
            .expect("bounded fixture position");
        assert_eq!(completed.positions[ordinal], position);
        // These fixtures cover no processors or dyadic frequency/presence.
        // Do not turn that restricted arithmetic proof into a generic claim
        // about repetition/min-p or subnormal CPU/GPU Stage equivalence.
        assert_eq!(request.processors.repetition_penalty, 1.0);
        assert_eq!(request.processors.min_p, 0.0);
        assert!(request.processors.logit_bias.is_empty());
        for (&raw, &count) in row.iter().zip(&counts) {
            let normal = |x: f32| x.is_finite() && (x == 0.0 || x.abs() >= f32::MIN_POSITIVE);
            assert!(
                normal(raw),
                "fixture requires normal finite raw logits or zero"
            );
            let frequency = count as f32 * request.processors.frequency_penalty as f32;
            let presence = if count > 0 {
                request.processors.presence_penalty as f32
            } else {
                0.0
            };
            assert!(normal(raw - frequency));
            assert!(normal((raw - frequency) - presence));
        }
        sampler.at_position(Some(position as u64));
        let token =
            blocking.sample_and_feed(&mut sampler, &Tensor::new(row.as_slice(), &Device::Cpu)?)?;
        let id = token.to_vec2::<u32>()?[0][0];
        counts[id as usize] += 1;
        expected.push(id);
    }
    assert_eq!(blocking.drain_all()?, expected);
    Ok(expected)
}

fn captured_ring_arms() -> [ReadbackArm; 4] {
    [
        ReadbackArm::Ring(crate::RingArm {
            lookahead: 0,
            commit_steps: 1,
            ..crate::RingArm::FIXED
        }),
        ReadbackArm::RING_FIXED,
        ReadbackArm::CROWN,
        ReadbackArm::Ring(crate::RingArm {
            eos: lloom::generation::EosRead::HostOnDrain,
            ..crate::RingArm::CROWN
        }),
    ]
}

#[test]
#[ignore = "explicit native admission; release+profile; one process per Metal API"]
fn prepared_ring_replays_same_raw_rows_on_cpu_for_original_sampling_arms() -> Result<()> {
    let mut decode = crate::DecodeConfig::default();
    decode.readback.device_filter = true;
    decode.readback.run_ahead = 3;
    decode.readback.queued_steps = 2;
    let (_files, engine) = fixture(decode)?;
    let prompt = engine.encode("Hello world", false)?;
    let vocab = engine.core.model.vocab_size();
    for gumbel in [false, true] {
        for filtered in [false, true] {
            for history in [false, true] {
                for arm in captured_ring_arms() {
                    let request = DecodeParams {
                        seed: 781,
                        temperature: Some(0.7),
                        gumbel,
                        top_k: filtered.then_some(8),
                        top_p: filtered.then_some(0.8),
                        processors: lloom::generation::Processors {
                            presence_penalty: if history { 0.5 } else { 0.0 },
                            frequency_penalty: if history { 0.25 } else { 0.0 },
                            ..Default::default()
                        },
                        max_tokens: 32,
                        eos_ids: NEVER_EOS.to_vec(),
                        readback: arm,
                        cold: true,
                        ..Default::default()
                    };
                    let original_sampling = request.sampling()?;
                    let capture = crate::decode::ring_capture::Capture::new(
                        &engine.core.device,
                        prompt.len(),
                        request.max_tokens,
                        vocab,
                    )?;
                    let census = engine
                        .core
                        .device
                        .as_metal_device()?
                        .dispatch_census()
                        .open()
                        .expect("isolated native census");
                    let result = captured_run(&engine, &prompt, &request, &capture);
                    let labels = census.close();
                    let (out, rows) = result?;
                    let expected = replay_captured_on_cpu(&prompt, &request, &rows)?;
                    assert_eq!(
                        out.sampling, original_sampling,
                        "do not normalize the sampling arm"
                    );
                    assert_eq!(
                        out.tokens, expected,
                        "gumbel={gumbel} filtered={filtered} history={history} arm={arm:?}"
                    );
                    let method = if gumbel {
                        "sampling_filtered_gumbel"
                    } else {
                        "sampling_multinomial"
                    };
                    assert!(
                        labels.iter().any(|(n, k)| n.contains(method) && *k > 0),
                        "native contract route absent: {labels:?}"
                    );
                    assert!(
                        !labels
                            .iter()
                            .any(|(n, k)| n.contains("sample_inverse_cdf") && *k > 0),
                        "incumbent bypass: {labels:?}"
                    );
                    println!("SAME_LOGITS_CPU original={original_sampling:?} history={history} arm={arm:?} positions={:?} ids={:?}", rows.positions, out.tokens);
                }
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "explicit native admission; release+profile; one process per Metal API"]
fn prepared_ring_centered_gumbel_and_forced_fed_history_have_causal_cpu_controls() -> Result<()> {
    let mut decode = crate::DecodeConfig::default();
    decode.readback.device_filter = true;
    let (_files, engine) = fixture(decode)?;
    let prompt = engine.encode("Hello world", false)?;
    let vocab = engine.core.model.vocab_size();
    assert!(vocab > 2);
    let n = 8;
    let request = DecodeParams {
        seed: 781,
        temperature: Some(3.0),
        gumbel: true,
        max_tokens: n,
        eos_ids: NEVER_EOS.to_vec(),
        readback: ReadbackArm::CROWN,
        cold: true,
        ..Default::default()
    };
    let scores = Tensor::full(1e30f32, (n, vocab), &engine.core.device)?;
    let capture =
        crate::decode::ring_capture::Capture::new(&engine.core.device, prompt.len(), n, vocab)?
            .with_scores(scores)?;
    let (out, rows) = captured_run(&engine, &prompt, &request, &capture)?;
    let expected = replay_captured_on_cpu(&prompt, &request, &rows)?;
    assert_ne!(
        expected,
        vec![0; n],
        "positive control must distinguish incumbent z+T*g low-ID tie"
    );
    assert_eq!(out.tokens, expected);
    for gumbel in [false, true] {
        let mut values = vec![-10_000f32; n * vocab];
        for row in 0..n {
            values[row * vocab + 1] = 0.0;
            if row > 0 {
                values[row * vocab + 2] = 0.25;
            }
        }
        let budget = lloom::generation::ThinkingBudget::new(0, vec![2], vec![2])?;
        let request = DecodeParams {
            gumbel,
            temperature: Some(0.001),
            processors: lloom::generation::Processors {
                frequency_penalty: 0.5,
                ..Default::default()
            },
            thinking: Some(budget),
            ..request.clone()
        };
        let scores = Tensor::from_vec(values.clone(), (n, vocab), &engine.core.device)?;
        let capture =
            crate::decode::ring_capture::Capture::new(&engine.core.device, prompt.len(), n, vocab)?
                .with_scores(scores)?;
        let (out, rows) = captured_run(&engine, &prompt, &request, &capture)?;
        let expected = replay_captured_on_cpu(&prompt, &request, &rows)?;
        assert_eq!(
            &expected[..2],
            &[2, 1],
            "forced id2, then id1 from fed-id2 device history"
        );
        assert_eq!(out.tokens, expected);
        let free = DecodeParams {
            thinking: None,
            ..request.clone()
        };
        let scores = Tensor::from_vec(values, (n, vocab), &engine.core.device)?;
        let capture =
            crate::decode::ring_capture::Capture::new(&engine.core.device, prompt.len(), n, vocab)?
                .with_scores(scores)?;
        let (next, rows) = captured_run(&engine, &prompt, &free, &capture)?;
        let expected = replay_captured_on_cpu(&prompt, &free, &rows)?;
        assert_eq!(expected[0], 1, "independent no-budget positive control");
        assert_eq!(
            next.tokens, expected,
            "previous budget cannot leak into the next request"
        );
        assert!(next.thinking.is_none());
    }
    Ok(())
}

#[test]
#[ignore = "explicit native admission; one process per Metal API"]
fn prepared_ring_snapshot_bytes_survive_reused_source_storage() -> Result<()> {
    let (_files, engine) = fixture(crate::DecodeConfig::default())?;
    let device = &engine.core.device;
    let capture = crate::decode::ring_capture::Capture::new(device, 7, 2, 3)?;
    let source = Tensor::new(&[1f32, 2., 3.], device)?;
    capture.record(&source, Some(7))?;
    source.slice_set(&Tensor::new(&[4f32, 5., 6.], device)?, 0, 0)?;
    capture.record(&source, Some(8))?;
    let overflow = capture.record(&source, Some(9));
    let result = capture.finish()?;
    assert!(overflow.is_err());
    result.check_matrix(7, 2, 3)?;
    assert_eq!(result.rows, [vec![1., 2., 3.], vec![4., 5., 6.]]);
    assert_eq!(source.to_vec1::<f32>()?, [4., 5., 6.]);
    Ok(())
}

#[test]
#[ignore = "explicit native admission; release+profile; one process per Metal API"]
fn prepared_ring_capture_rejects_a_shifted_actual_sampler_key() -> Result<()> {
    let mut decode = crate::DecodeConfig::default();
    decode.readback.device_filter = true;
    let (_files, engine) = fixture(decode)?;
    let prompt = engine.encode("Hello world", false)?;
    let request = DecodeParams {
        seed: 781,
        temperature: Some(0.7),
        gumbel: true,
        max_tokens: 8,
        eos_ids: NEVER_EOS.to_vec(),
        readback: ReadbackArm::CROWN,
        cold: true,
        ..Default::default()
    };
    let make = || {
        crate::decode::ring_capture::Capture::new(
            &engine.core.device,
            prompt.len(),
            request.max_tokens,
            engine.core.model.vocab_size(),
        )
    };
    let positive = make()?;
    let (out, rows) = captured_run(&engine, &prompt, &request, &positive)?;
    assert_eq!(
        out.tokens,
        replay_captured_on_cpu(&prompt, &request, &rows)?
    );
    let negative = make()?.shifted_first_key_control();
    match captured_run(&engine, &prompt, &request, &negative) {
        Err(error)
            if error
                .to_string()
                .contains("missing, duplicate, shifted or malformed row receipt") => {}
        Err(error) => return Err(error),
        Ok(_) => anyhow::bail!("actual shifted sampler key escaped the independent position check"),
    }
    // This is the real sampler seam's key, observed before delegated native
    // sampling, not a manually corrupted Completed receipt.
    let observed = negative.observed_positions();
    assert_eq!(observed[0], prompt.len() + 1);
    assert_eq!(observed[1], prompt.len() + 1);
    println!("SAME_LOGITS_SHIFTED_KEY rejected=true observed={observed:?}");
    Ok(())
}

use std::io::Write as _;

// Diagnostic-only fixture observer. No logits copies or extra GPU dispatches.
// Existing DeviceTrace is reused; its host allocations/locks forbid timing use.
struct BodyObserver {
    frames: std::cell::RefCell<Vec<BodyFrame>>,
    capacity: usize,
    lost: std::cell::Cell<bool>,
}

struct BodyFrame {
    route: &'static str,
    pos: usize,
    shape: Vec<usize>,
    plan: Option<serde_json::Value>,
    outcome: String,
    dispatches: Vec<lloom_tensor::kernels::metal::trace::Dispatch>,
}

struct BodyScope<'a> {
    owner: &'a BodyObserver,
    frame: Option<BodyFrame>,
    window: Option<lloom_tensor::kernels::metal::trace::TraceWindow>,
}

impl Drop for BodyScope<'_> {
    fn drop(&mut self) {
        // Preserve the attempted trace even if the delegated body unwinds.
        // The existing window closes before any later body may open a window.
        if let (Some(mut frame), Some(window)) = (self.frame.take(), self.window.take()) {
            frame.dispatches = window.close();
            if let Ok(mut frames) = self.owner.frames.try_borrow_mut() {
                if frames.len() < self.owner.capacity {
                    frames.push(frame);
                    return;
                }
            }
            self.owner.lost.set(true);
        }
    }
}

impl BodyObserver {
    fn new(capacity: usize) -> Self {
        Self {
            frames: std::cell::RefCell::new(Vec::with_capacity(capacity)),
            capacity,
            lost: std::cell::Cell::new(false),
        }
    }

    fn observe(
        &self,
        route: &'static str,
        input: &Tensor,
        pos: usize,
        body: impl FnOnce() -> lloom_tensor::Result<Tensor>,
    ) -> lloom_tensor::Result<Tensor> {
        if self.frames.borrow().len() >= self.capacity {
            self.lost.set(true);
            lloom_tensor::bail!("body observer capacity exhausted; no complete witness");
        }
        let metal = input.device().as_metal_device()?;
        // Read installed metadata inside the pass, before end_pass clears it.
        // Keep only plain host metadata: do not retain the arena's allocation.
        let plan = metal.arena().map(|arena| {
            serde_json::json!({
                "base_addr": arena.base().raw_addr(),
                "base_offset": arena.base().base_offset(),
                "base_bytes": arena.base().length(),
                "served_counter_inside_body": metal.pass_plan_counters().map(|c| c.served),
                "plan_debug": format!("{:?}", arena.plan_snapshot()),
                "layout": format!("{:?}", arena.layout()),
                "offsets": format!("{:?}", arena.offsets()),
            })
        });
        let window = metal
            .dispatch_trace()
            .open()
            .map_err(|e| lloom_tensor::Error::Msg(format!("body observer trace refused: {e:?}")))?;
        let mut scope = BodyScope {
            owner: self,
            frame: Some(BodyFrame {
                route,
                pos,
                shape: input.dims().to_vec(),
                plan,
                outcome: "unwound".into(),
                dispatches: Vec::new(),
            }),
            window: Some(window),
        };
        let result = body(); // exactly one call, unchanged output Tensor
        scope.frame.as_mut().unwrap().outcome = match &result {
            Ok(_) => "returned".into(),
            Err(error) => format!("error: {error}"),
        };
        drop(scope);
        result
    }

    fn dump(
        &self,
        output: &mut impl std::io::Write,
        arm: &str,
        terminal: &str,
    ) -> std::io::Result<()> {
        // All JSON/dispatch formatting happens after the request's data fence.
        for (index, frame) in self.frames.borrow().iter().enumerate() {
            let dispatches: Vec<_> = frame
                .dispatches
                .iter()
                .map(|d| {
                    serde_json::json!({
                        "seq": d.seq, "pipeline": d.pipeline,
                        "grid": d.grid, "threadgroup": d.threadgroup,
                        "by_threadgroups": d.by_threadgroups,
                        "barrier": d.barrier, "kinds": format!("{:?}", d.kinds),
                        "encoder": d.encoder, "region": d.region,
                        "bindings": d.bindings.iter().map(|b| serde_json::json!({
                            "index": b.index, "buffer_id": b.buffer_id,
                            "buffer_addr": b.buffer_addr, "offset": b.offset,
                            "len": b.len, "is_output": b.is_output,
                        })).collect::<Vec<_>>(),
                    })
                })
                .collect();
            writeln!(
                output,
                "BODY_TRACE {}",
                serde_json::json!({
                    "arm": arm, "frame": index, "route": frame.route,
                    "input_pos": frame.pos, "input_shape": frame.shape,
                    "plan": frame.plan, "outcome": frame.outcome,
                    "terminal": terminal, "lost": self.lost.get(),
                    "coordinate": "trace seq, NOT STEP_DISPATCH",
                    "dispatches": dispatches,
                })
            )?;
        }
        Ok(())
    }

    fn check_frames(&self, prompt: usize, n: usize, out: &Decoded, prepared: bool) -> Result<()> {
        anyhow::ensure!(!self.lost.get(), "body trace lost a frame");
        anyhow::ensure!(
            !out.stats.provisional && out.stats.redraws == 0,
            "unexpected replay"
        );
        anyhow::ensure!(
            out.stop == Stop::Length && out.tokens.len() == n,
            "early stop"
        );
        let frames = self.frames.borrow();
        let u = out.stats.prefill_units;
        anyhow::ensure!(
            u > 0 && u <= prompt && frames.len() == u + n,
            "U+N body count"
        );
        let mut pos = 0usize;
        for frame in &frames[..u] {
            anyhow::ensure!(
                frame.route == "token" && frame.pos == pos,
                "prefill position/route"
            );
            anyhow::ensure!(
                frame.shape.len() == 2 && frame.shape[0] == 1,
                "prefill input shape"
            );
            let width = frame.shape[1];
            anyhow::ensure!(width > 0 && width <= prompt - pos, "prefill span");
            pos += width;
        }
        anyhow::ensure!(pos == prompt, "prefill coverage");
        for (i, frame) in frames[u..].iter().enumerate() {
            anyhow::ensure!(frame.pos == prompt + i, "decode input position");
            if prepared {
                anyhow::ensure!(
                    frame.route == "prepared"
                        && frame.shape.len() == 3
                        && frame.shape[0] == 1
                        && frame.shape[1] == 1,
                    "prepared input route"
                );
            } else {
                anyhow::ensure!(
                    frame.route == "token" && frame.shape == [1, 1],
                    "ordinary input route"
                );
            }
        }
        anyhow::ensure!(
            frames
                .iter()
                .all(|f| f.outcome == "returned" && !f.dispatches.is_empty()),
            "failed or empty body trace"
        );
        // A trace is not an alias verdict; later semantic binding analysis must
        // establish the specific residual/up/down edges within each body.
        Ok(())
    }
}

// First tiny LFM2 conv layer: forward_add produces R, MLP retains R while
// gate/up/product run, down reads R at argument15. See models/lfm2.rs
// DecoderLayer::forward, ShortConv::forward_add and MLP::{gated,forward_add}.
// Logical extents are H128/F16=256 and I256/F16=512, NOT slot handle lengths.
struct FirstConvEdge<'a> {
    residual_dispatch: &'a lloom_tensor::kernels::metal::trace::Dispatch,
    up_dispatch: &'a lloom_tensor::kernels::metal::trace::Dispatch,
    down_dispatch: &'a lloom_tensor::kernels::metal::trace::Dispatch,
    residual: &'a lloom_tensor::kernels::metal::trace::Binding,
    up: &'a lloom_tensor::kernels::metal::trace::Binding,
    down_residual: &'a lloom_tensor::kernels::metal::trace::Binding,
    overlap: bool,
}

fn explicit_edge_binding(
    dispatch: &lloom_tensor::kernels::metal::trace::Dispatch,
    index: usize,
    output: bool,
    logical_bytes: usize,
) -> Result<&lloom_tensor::kernels::metal::trace::Binding> {
    let mut candidates = dispatch.bindings.iter().filter(|b| b.index == index);
    let b = candidates.next().ok_or_else(|| {
        anyhow::anyhow!("model edge witness incomplete: missing explicit argument {index}")
    })?;
    anyhow::ensure!(
        candidates.next().is_none()
            && b.is_output == output
            && b.buffer_addr != 0
            && b.len >= logical_bytes
            && b.offset.checked_add(b.len).is_some(),
        "model edge witness incomplete: duplicate, unknown or short argument {index}"
    );
    Ok(b)
}

fn first_conv_edge(
    dispatches: &[lloom_tensor::kernels::metal::trace::Dispatch],
) -> Result<FirstConvEdge<'_>> {
    let first = dispatches
        .iter()
        .position(|d| d.pipeline == "shortconv_step_f16_k3")
        .ok_or_else(|| anyhow::anyhow!("model edge witness incomplete: first shortconv route"))?;
    let d = dispatches
        .get(first..first + 7)
        .ok_or_else(|| anyhow::anyhow!("model edge witness incomplete: first MLP sequence"))?;
    let gemv = "gemv_float16_bm4_bn1_sm1_sn32_tm4_tn4_nc0_axpby0";
    anyhow::ensure!(
        d[1].pipeline == format!("{gemv}_residual")
            && d[2].pipeline == "rmsnorm_f16"
            && d[3].pipeline == format!("{gemv}_silu")
            && d[4].pipeline == gemv
            && d[5].pipeline == "bmul_f16"
            && d[6].pipeline == format!("{gemv}_residual")
            && d.windows(2).all(|w| w[0].seq < w[1].seq),
        "model edge witness incomplete: changed conv/norm/gate/up/product/down route"
    );
    let residual = explicit_edge_binding(&d[1], 3, true, 256)?;
    let up = explicit_edge_binding(&d[4], 3, true, 512)?;
    let down_residual = explicit_edge_binding(&d[6], 15, false, 256)?;
    anyhow::ensure!(
        residual.buffer_addr == down_residual.buffer_addr
            && residual.offset == down_residual.offset,
        "model edge witness incomplete: down does not read original residual"
    );
    // Object identity + within-object byte offsets; NOT GPU virtual addresses.
    let overlap = residual.buffer_addr == up.buffer_addr
        && residual.offset < up.offset + 512
        && up.offset < residual.offset + 256;
    Ok(FirstConvEdge {
        residual_dispatch: &d[1],
        up_dispatch: &d[4],
        down_dispatch: &d[6],
        residual,
        up,
        down_residual,
        overlap,
    })
}

fn edge_binding_json(
    b: &lloom_tensor::kernels::metal::trace::Binding,
    logical_bytes: usize,
) -> serde_json::Value {
    serde_json::json!({"argument": b.index, "output": b.is_output,
        "buffer_object": b.buffer_addr, "interned_id": b.buffer_id,
        "offset": b.offset, "logical_bytes": logical_bytes,
        "logical_end": b.offset + logical_bytes, "handle_bytes": b.len})
}

impl BodyObserver {
    fn check_first_conv_edges(&self, arm: &str, prefill_units: usize) -> Result<()> {
        let frames = self.frames.borrow();
        let mut installed_bodies = 0usize;
        for (index, frame) in frames.iter().enumerate().skip(prefill_units) {
            let edge = first_conv_edge(&frame.dispatches)?;
            let witness = serde_json::json!({
                "arm": arm, "frame": index, "input_pos": frame.pos, "route": frame.route,
                "r": edge_binding_json(edge.residual, 256),
                "up": edge_binding_json(edge.up, 512),
                "down_residual": edge_binding_json(edge.down_residual, 256),
                "pipelines": [edge.residual_dispatch.pipeline, edge.up_dispatch.pipeline,
                    edge.down_dispatch.pipeline],
                "trace_seq": [edge.residual_dispatch.seq, edge.up_dispatch.seq, edge.down_dispatch.seq],
                "coordinate": "trace seq, NOT STEP_DISPATCH; MTLBuffer object, NOT GPU VA",
                "plan": frame.plan, "overlap": edge.overlap,
            });
            writeln!(std::io::stdout().lock(), "BODY_MODEL_EDGE {witness}")?;
            if edge.overlap {
                writeln!(std::io::stdout().lock(), "BODY_ALIAS_CONFLICT {witness}")?;
                anyhow::bail!("model residual/up alias before down read");
            }
            if let Some(plan) = &frame.plan {
                let base = plan["base_addr"].as_u64().ok_or_else(|| {
                    anyhow::anyhow!("model edge witness incomplete: installed base identity")
                })?;
                anyhow::ensure!(
                    plan["served_counter_inside_body"]
                        .as_u64()
                        .is_some_and(|n| n > 0),
                    "model edge witness incomplete: no serving counter inside installed body"
                );
                // A body actually bound an operand to its installed arena,
                // stronger than finding a key in the cache AFTER the request.
                if [edge.residual, edge.up, edge.down_residual]
                    .into_iter()
                    .any(|b| b.buffer_addr as u64 == base)
                {
                    installed_bodies += 1;
                }
            }
        }
        anyhow::ensure!(
            installed_bodies > 0,
            "model edge witness incomplete: route executed no arena-bound decode body"
        );
        writeln!(std::io::stdout().lock(), "BODY_PLAN_SERVED arm={arm} arena_bound_bodies={installed_bodies} active_key_event=UNOBSERVED served_offer_ordinals=UNOBSERVED")?;
        Ok(())
    }
}

#[cfg(test)]
mod model_edge_controls {
    use super::*;
    use lloom_tensor::kernels::metal::trace::{Binding, Dispatch};

    fn binding(index: usize, output: bool, offset: usize, len: usize) -> Binding {
        Binding {
            index,
            is_output: output,
            buffer_addr: 42,
            buffer_id: 2,
            offset,
            len,
        }
    }

    fn example(up_offset: usize) -> Vec<Dispatch> {
        let gemv = "gemv_float16_bm4_bn1_sm1_sn32_tm4_tn4_nc0_axpby0";
        [
            "shortconv_step_f16_k3".into(),
            format!("{gemv}_residual"),
            "rmsnorm_f16".into(),
            format!("{gemv}_silu"),
            gemv.into(),
            "bmul_f16".into(),
            format!("{gemv}_residual"),
        ]
        .into_iter()
        .enumerate()
        .map(|(i, pipeline)| Dispatch {
            seq: i as u64,
            pipeline,
            grid: (1, 1, 1),
            threadgroup: (1, 1, 1),
            by_threadgroups: true,
            bindings: match i {
                1 => vec![binding(3, true, 2304, 512)],
                4 => vec![binding(3, true, up_offset, 512)],
                6 => vec![binding(15, false, 2304, 512)],
                _ => Vec::new(),
            },
            barrier: false,
            kinds: lloom_tensor::kernels::metal::encoder::HazardKinds::NONE,
            encoder: 1,
            region: None,
        })
        .collect()
    }

    #[test]
    fn actual_model_edge_validator_accepts_adjacent_logical_ranges() {
        // R's handle is512 but logical R256: adjacency at2560 is not overlap.
        assert!(!first_conv_edge(&example(2560)).unwrap().overlap);
    }

    #[test]
    fn actual_model_edge_validator_detects_observed_and_partial_overlap() {
        assert!(first_conv_edge(&example(2304)).unwrap().overlap);
        assert!(first_conv_edge(&example(2048)).unwrap().overlap);
    }

    #[test]
    fn actual_model_edge_validator_refuses_incomplete_or_changed_witness() {
        for case in 0..6 {
            let mut d = example(2560);
            match case {
                0 => d[4].bindings.clear(),
                1 => d[4].bindings.push(binding(3, true, 2560, 512)),
                2 => d[4].bindings[0].len = 0,
                3 => d[4].bindings[0].offset = usize::MAX,
                4 => d[6].bindings[0].offset += 256,
                5 => d[4].pipeline = "unrecognized route".into(),
                _ => unreachable!(),
            }
            assert!(
                first_conv_edge(&d).is_err(),
                "incomplete case {case} accepted"
            );
        }
    }
}

// Diagnostics are secondary to the delegated body and its data fence.
fn finish_body_observation<T>(
    outcome: std::thread::Result<lloom_tensor::Result<T>>,
    terminal: lloom_tensor::Result<()>,
    dump: impl FnOnce() -> std::io::Result<()>,
    secondary: &mut impl std::io::Write,
) -> Result<T> {
    let diagnostic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(dump))
        .unwrap_or_else(|_| Err(std::io::Error::other("body trace dump panicked")));
    // A failing diagnostic sink must not replace the body or fence failure.
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if let Err(error) = &terminal {
            let _ = writeln!(secondary, "BODY_TRACE_CLEANUP_FAILED {error}");
        }
        if let Err(error) = &diagnostic {
            let _ = writeln!(secondary, "BODY_TRACE_OUTPUT_FAILED {error}");
        }
    }));
    match outcome {
        Err(payload) => std::panic::resume_unwind(payload),
        Ok(Err(error)) => Err(error.into()),
        Ok(Ok(value)) => {
            terminal?;
            diagnostic?;
            Ok(value)
        }
    }
}

#[test]
#[ignore = "explicit diagnostic admission; release+profile; no performance claim"]
fn prepared_then_ordinary_same_owner_exports_body_plans_and_bindings() -> Result<()> {
    // Preserve the original warm-prefix reproduction and strict cold validator:
    // its archived ordinary-prefix failure is not rewritten into a pass.
    observe_same_owner_body_input(false, false)
}

#[test]
#[ignore = "separate cold-reset diagnostic admission; release+profile; not performance"]
fn prepared_then_ordinary_cold_reset_exports_body_plans_and_bindings() -> Result<()> {
    observe_same_owner_body_input(true, false)
}

#[test]
#[ignore = "separate reverse cold-reset diagnostic; release+profile; not performance"]
fn ordinary_then_prepared_cold_reset_exports_body_plans_and_bindings() -> Result<()> {
    observe_same_owner_body_input(true, true)
}

fn observe_same_owner_body_input(reset_each_request: bool, ordinary_first: bool) -> Result<()> {
    let mut decode = crate::DecodeConfig::default();
    decode.readback.device_filter = true;
    decode.readback.run_ahead = 3;
    decode.readback.queued_steps = 2;
    let (_files, engine) = fixture_with_trace(decode, true)?;
    anyhow::ensure!(
        lloom_tensor::kernels::metal::trace::trace_requested(),
        "trace not installed"
    );
    let prompt = engine.encode("Hello world", false)?;
    let n = 8usize;
    anyhow::ensure!(
        engine.core.model.vocab_size() > 98 && !prompt.is_empty(),
        "fixture geometry"
    );
    let mut forced = vec![97; n];
    forced[n - 1] = 98;
    let budget = lloom::generation::ThinkingBudget::new(0, vec![98], forced.clone())?;
    let arms = if ordinary_first {
        [
            ("ordinary-first", ReadbackArm::Blocking, false),
            ("native-after-ordinary", ReadbackArm::CROWN, true),
        ]
    } else {
        [
            ("native-first", ReadbackArm::CROWN, true),
            ("ordinary-after-native", ReadbackArm::Blocking, false),
        ]
    };
    let metal = engine.core.device.as_metal_device()?;
    let mut key_assertions = Vec::new();
    for (arm_index, (label, arm, prepared)) in arms.into_iter().enumerate() {
        let served_before = metal.pass_plan_counters().map(|c| c.served);
        let observer = BodyObserver::new(prompt.len().checked_add(n).unwrap());
        let model = InvalidLogits {
            inner: &engine.core.model,
            from: usize::MAX,
            panic: false,
            injected: AtomicUsize::new(0),
            prepared: AtomicUsize::new(0),
            observer: Some(&observer),
        };
        let request = DecodeParams {
            max_tokens: n,
            thinking: Some(budget.clone()),
            ..params(arm, false, true)
        };
        let mut lease = engine.checkout_single()?;
        let config = lease.extra().decode.clone();
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            lease.with_model(&model, &engine.core.device, config, |session, _| {
                // The public Engine::complete_until applies its cold flag at
                // this layer; direct with_model callers otherwise keep prefix state.
                // Existing Session::reset does not clear the device's pass plans.
                if reset_each_request {
                    session.reset();
                }
                session.decode_until(&prompt, engine.detok(), &request, |_| true)
            })
        }));
        // Keep owner custody through the diagnostic data fence, including a
        // failed/unwound body. No assertion runs before this cleanup attempt.
        let terminal = engine.core.device.synchronize();
        let terminal_note = match &terminal {
            Ok(()) => "synchronized".to_owned(),
            Err(error) => format!("synchronize failed: {error}"),
        };
        let decoded = finish_body_observation(
            outcome,
            terminal,
            || observer.dump(&mut std::io::stdout().lock(), label, &terminal_note),
            &mut std::io::stderr().lock(),
        )?;
        lease.mark_complete();
        drop(lease);
        anyhow::ensure!(
            decoded.tokens == forced,
            "actual fed IDs differ from forced schedule"
        );
        anyhow::ensure!(
            model.injected.load(Ordering::Relaxed) == 0,
            "unexpected corruption"
        );
        anyhow::ensure!(
            model.prepared.load(Ordering::Relaxed) == if prepared { n } else { 0 },
            "prepared forward count"
        );
        writeln!(
            std::io::stdout().lock(),
            "BODY_TRACE_PREPARATION {}",
            serde_json::json!({
                "arm": label, "cold_asked": request.cold,
                "explicit_session_reset": reset_each_request,
                "reused": decoded.reused, "prefilled": decoded.prefilled,
                "reuse": format!("{:?}", decoded.reuse),
                "turn": decoded.turn.json(Some(engine.reuse_prefix)),
                "checkpoint": format!("{:?}", decoded.checkpoint),
                "checkpoint_at": decoded.checkpoint_at,
            })
        )?;
        if reset_each_request {
            anyhow::ensure!(
                decoded.reused == 0 && decoded.prefilled == prompt.len(),
                "cold-reset request reused a prefix"
            );
            anyhow::ensure!(
                decoded.turn.boundary == 0
                    && decoded.turn.overrun == 0
                    && decoded.turn.reused == 0
                    && decoded.turn.replayed == 0
                    && decoded.turn.prefilled == prompt.len()
                    && decoded.turn.outcome == crate::turn::Outcome::Fresh
                    && decoded.turn.reason == crate::turn::Reason::Cold
                    && decoded.turn.state == crate::turn::StateReady::Exact,
                "cold-reset turn preparation differs"
            );
        }
        // Keep exact start0/full prompt coverage; no cross-prefill-split
        // numeric identity is inferred from this control or forced output IDs.
        observer.check_frames(prompt.len(), n, &decoded, prepared)?;
        if reset_each_request {
            // Physical lifetime check precedes key-label checks: the collapse
            // mutant must reproduce the model edge, not just a missing label.
            observer.check_first_conv_edges(label, decoded.stats.prefill_units)?;
            let served_after = metal.pass_plan_counters().map(|c| c.served);
            anyhow::ensure!(
                matches!((served_before, served_after), (Some(a), Some(b)) if b > a),
                "input-route witness incomplete: no cached pass served this request"
            );
            let (entries, base_bytes) = metal
                .pass_plan_report()
                .ok_or_else(|| anyhow::anyhow!("input-route witness incomplete: no plan cache"))?;
            let expected = if prepared {
                lloom_tensor::backend::PassInput::PreparedEmbeddings
            } else {
                lloom_tensor::backend::PassInput::TokenIds
            };
            let decode_keys: Vec<_> = entries
                .iter()
                .filter(|(k, ..)| {
                    k.kind == (lloom_tensor::kernels::metal::arena::PassKind::Decode { b: 1 })
                        && k.opener == lloom_tensor::kernels::metal::arena::PassOpener::Driver
                })
                .collect();
            writeln!(
                std::io::stdout().lock(),
                "BODY_PLAN_KEYS {}",
                serde_json::json!({
                    "arm": label, "expected_input": format!("{expected:?}"),
                    "cached_keys": decode_keys.iter().map(|(k, covered, total, escapes)|
                        serde_json::json!({"key": format!("{k:?}"), "input": format!("{:?}", k.input),
                            "variant": k.variant, "arms": k.arms, "covered": covered,
                            "total": total, "escapes": escapes})).collect::<Vec<_>>(),
                    "base_bytes": base_bytes, "served_before": served_before, "served_after": served_after,
                    "custody": "cached key list; NOT active-key or served-offer ordinal event",
                })
            )?;
            // Retain this arm's exact snapshot, but defer key-label verdicts
            // until BOTH physical paths ran. Otherwise a collapse mutant dies
            // on the native-first label before it can encode ordinary's alias.
            key_assertions.push((
                label,
                "expected input-route key absent",
                decode_keys
                    .iter()
                    .any(|(k, ..)| k.input == expected && k.variant == 0),
            ));
            if arm_index == 1 {
                key_assertions.push((
                    label,
                    "both input-route keys must coexist",
                    [
                        lloom_tensor::backend::PassInput::TokenIds,
                        lloom_tensor::backend::PassInput::PreparedEmbeddings,
                    ]
                    .into_iter()
                    .all(|input| {
                        decode_keys
                            .iter()
                            .any(|(k, ..)| k.input == input && k.variant == 0)
                    }),
                ));
            }
        }
        let alias_verdict = if reset_each_request {
            "FIRST_CONV_EDGE_CHECKED"
        } else {
            "UNASSESSED"
        };
        writeln!(std::io::stdout().lock(), "BODY_TRACE_COMPLETE arm={label} inputs={:?} output_ids={:?} frames={} alias_verdict={alias_verdict}",
            prompt, decoded.tokens, observer.frames.borrow().len())?;
    }
    for (label, message, satisfied) in key_assertions {
        anyhow::ensure!(satisfied, "{label}: {message}");
    }
    Ok(())
}

// Proposed host-only controls: no fixture/device/Metal operation.
#[cfg(test)]
mod body_trace_output_controls {
    use super::*;

    struct FailedOutput;
    impl std::io::Write for FailedOutput {
        fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
            Err(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "injected output failure",
            ))
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    fn observer() -> BodyObserver {
        let observer = BodyObserver::new(1);
        observer.frames.borrow_mut().push(BodyFrame {
            route: "host-control",
            pos: 0,
            shape: vec![1, 1],
            plan: None,
            outcome: "returned".into(),
            dispatches: Vec::new(),
        });
        observer
    }

    #[test]
    fn successful_dump_has_a_positive_output_control() {
        let observer = observer();
        let mut output = Vec::new();
        let value = finish_body_observation(
            Ok(Ok(7)),
            Ok(()),
            || observer.dump(&mut output, "host", "synchronized"),
            &mut Vec::new(),
        )
        .unwrap();
        assert_eq!(value, 7);
        let text = String::from_utf8(output).unwrap();
        assert!(text.starts_with("BODY_TRACE "));
        assert!(text.contains("host-control"));
    }

    #[test]
    fn failed_output_preserves_decode_error_and_cleanup_report() {
        let observer = observer();
        let mut secondary = Vec::new();
        let result = finish_body_observation::<()>(
            Ok(Err(lloom_tensor::Error::Msg("primary decode error".into()))),
            Err(lloom_tensor::Error::Msg("secondary fence error".into())),
            || observer.dump(&mut FailedOutput, "host", "fence failed"),
            &mut secondary,
        );
        assert_eq!(result.unwrap_err().to_string(), "primary decode error");
        let text = String::from_utf8(secondary).unwrap();
        assert!(text.contains("BODY_TRACE_CLEANUP_FAILED secondary fence error"));
        assert!(text.contains("BODY_TRACE_OUTPUT_FAILED injected output failure"));
    }

    #[test]
    fn failed_output_preserves_decode_panic_payload() {
        let observer = observer();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            finish_body_observation::<()>(
                Err(Box::new(781usize)),
                Err(lloom_tensor::Error::Msg("secondary fence error".into())),
                || observer.dump(&mut FailedOutput, "host", "fence failed"),
                &mut FailedOutput,
            )
        }));
        let payload = result.unwrap_err();
        assert_eq!(
            *payload.downcast::<usize>().expect("original panic type"),
            781
        );
    }

    #[test]
    fn failed_output_preserves_fence_error_after_decode_success() {
        let observer = observer();
        let result = finish_body_observation(
            Ok(Ok(7)),
            Err(lloom_tensor::Error::Msg("primary fence error".into())),
            || observer.dump(&mut FailedOutput, "host", "fence failed"),
            &mut Vec::new(),
        );
        assert_eq!(result.unwrap_err().to_string(), "primary fence error");
    }

    #[test]
    fn output_failure_is_error_when_body_and_fence_succeed() {
        let observer = observer();
        let result = finish_body_observation(
            Ok(Ok(7)),
            Ok(()),
            || observer.dump(&mut FailedOutput, "host", "synchronized"),
            &mut Vec::new(),
        );
        assert_eq!(result.unwrap_err().to_string(), "injected output failure");
    }

    #[test]
    fn unexpected_diagnostic_panic_does_not_replace_decode_error() {
        let result = finish_body_observation::<()>(
            Ok(Err(lloom_tensor::Error::Msg("primary decode error".into()))),
            Ok(()),
            || panic!("injected diagnostic panic"),
            &mut FailedOutput,
        );
        assert_eq!(result.unwrap_err().to_string(), "primary decode error");
    }
}

// Additive successor: the historical cold/warm controls above stay unchanged.
// Actual loaded Engine/RequestLease owner; same Session implementation, no new loop.
const OWNER_WARM_PROMPT: [u32; 11] = [72, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100];
const OWNER_WARM_FED: [u32; 8] = [97, 97, 97, 97, 97, 97, 97, 98];

fn owner_reused_request(
    engine: &NativeEngine,
    prepared_input: bool,
    reset: bool,
    planned: bool,
    label: &str,
) -> Result<()> {
    let observer = BodyObserver::new(64);
    let model = InvalidLogits {
        inner: &engine.core.model,
        from: usize::MAX,
        panic: false,
        injected: AtomicUsize::new(0),
        prepared: AtomicUsize::new(0),
        observer: Some(&observer),
    };
    let arm = if prepared_input {
        ReadbackArm::CROWN
    } else {
        ReadbackArm::Blocking
    };
    let budget = lloom::generation::ThinkingBudget::new(0, vec![98], OWNER_WARM_FED.to_vec())?;
    let request = DecodeParams {
        max_tokens: OWNER_WARM_FED.len(),
        thinking: Some(budget),
        // Direct with_model deliberately retains the owner's sequence. Reset,
        // when requested, is explicit below, matching complete_until's cold seam.
        ..params(arm, false, reset)
    };
    let mut lease = engine.checkout_single()?;
    let config = lease.extra().decode.clone();
    let mut before = None;
    let mut after = None;
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        lease.with_model(&model, &engine.core.device, config, |session, _| {
            if reset {
                session.reset();
            }
            before = Some(session.pos());
            let result =
                session.decode_until(&OWNER_WARM_PROMPT, engine.detok(), &request, |_| true);
            // Capture before borrowed-model custody returns; never synthesize
            // the terminal token log from the request/expected output.
            after = Some((session.pos(), session.tokens().to_vec()));
            result
        })
    }));
    let terminal = engine.core.device.synchronize();
    let terminal_note = format!("{terminal:?}");
    let decoded = finish_body_observation(
        outcome,
        terminal,
        || observer.dump(&mut std::io::stdout().lock(), label, &terminal_note),
        &mut std::io::stderr().lock(),
    )?;
    // Match existing owner diagnostic custody: a successful decode/fence is
    // complete before a test assertion. Assertions cannot disguise a body error.
    lease.mark_complete();
    drop(lease);
    let expected_start = if reset { 0 } else { 10 };
    anyhow::ensure!(
        before == Some(if reset { 0 } else { 19 }),
        "owner warm initial frontier"
    );
    anyhow::ensure!(
        decoded.reused == expected_start
            && decoded.prefilled == OWNER_WARM_PROMPT.len() - expected_start
            && decoded.reuse
                == if reset {
                    lloom::nn::ReuseAction::Fresh
                } else {
                    lloom::nn::ReuseAction::Rewound(9)
                },
        "actual owner warm preparation: {decoded:?}"
    );
    anyhow::ensure!(
        decoded.turn.boundary == if reset { 0 } else { 19 }
            && decoded.turn.overrun == 0
            && decoded.turn.reused == expected_start
            && decoded.turn.replayed == 0
            && decoded.turn.prefilled == OWNER_WARM_PROMPT.len() - expected_start,
        "owner turn preparation disagrees with exact reused suffix"
    );
    anyhow::ensure!(
        after == Some((19, [&OWNER_WARM_PROMPT[..], &OWNER_WARM_FED[..]].concat())),
        "actual owner terminal position/token log"
    );
    anyhow::ensure!(
        decoded.tokens == OWNER_WARM_FED && decoded.stop == Stop::Length,
        "actual owner forced feed boundary"
    );
    anyhow::ensure!(
        !decoded.stats.provisional && decoded.stats.redraws == 0,
        "warm physical witness unexpectedly redrew"
    );
    anyhow::ensure!(
        model.injected.load(Ordering::Relaxed) == 0,
        "warm witness unexpectedly injected failure"
    );
    anyhow::ensure!(
        model.prepared.load(Ordering::Relaxed) == if prepared_input { 8 } else { 0 },
        "actual prepared model forward was not selected"
    );
    anyhow::ensure!(!observer.lost.get(), "warm observer lost frame");
    let frames = observer.frames.borrow();
    let units = decoded.stats.prefill_units;
    anyhow::ensure!(
        units == 1 && frames.len() == 9,
        "warm owner exact body population"
    );
    let first = &frames[0];
    anyhow::ensure!(
        first.route == "token"
            && first.pos == expected_start
            && first.shape == [1, OWNER_WARM_PROMPT.len() - expected_start],
        "preparation-derived owner prefill span/route"
    );
    for (i, frame) in frames[units..].iter().enumerate() {
        anyhow::ensure!(
            frame.pos == 11 + i
                && frame.route == if prepared_input { "prepared" } else { "token" }
                && frame.shape
                    == if prepared_input {
                        vec![1, 1, 128]
                    } else {
                        vec![1, 1]
                    },
            "warm owner decode route/position/shape"
        );
    }
    anyhow::ensure!(
        frames
            .iter()
            .all(|frame| frame.outcome == "returned" && !frame.dispatches.is_empty()),
        "warm owner failed or empty body"
    );
    if !planned {
        // Same physical R/up/down parser; no arena requirement for pool-only cold control.
        for (index, frame) in frames.iter().enumerate().skip(units) {
            let edge = first_conv_edge(&frame.dispatches)?;
            writeln!(
                std::io::stdout().lock(),
                "OWNER_COLD_MODEL_EDGE {}",
                serde_json::json!({
                    "arm": label, "frame": index, "input_pos": frame.pos, "route": frame.route,
                    "residual": edge_binding_json(edge.residual, 256),
                    "up": edge_binding_json(edge.up, 512),
                    "down_residual": edge_binding_json(edge.down_residual, 256),
                    "overlap": edge.overlap, "arena_required": false,
                })
            )?;
            anyhow::ensure!(
                !edge.overlap,
                "cold owner residual/up alias before down read"
            );
        }
    }
    drop(frames);
    if planned {
        observer.check_first_conv_edges(label, units)?;
    }
    // Suffix equality is determined by the actual engine preparation fields
    // and its unchanged prepared.suffix forward call; no extra GPU readback is
    // inserted into prefill to manufacture an input witness.
    writeln!(
        std::io::stdout().lock(),
        "OWNER_REUSED_PREPARATION {}",
        serde_json::json!({
            "arm": label, "before": before, "reused": decoded.reused,
            "suffix_from_request": &OWNER_WARM_PROMPT[decoded.reused..],
            "prefilled": decoded.prefilled, "reuse": format!("{:?}", decoded.reuse),
            "turn": decoded.turn.json(Some(true)), "body_first_pos": expected_start,
            "forced_feed_only": true, "numerical_cross_split_claim": false,
        })
    )?;
    Ok(())
}

fn owner_reused_order(prepared_first: bool) -> Result<()> {
    let mut decode = crate::DecodeConfig::default();
    decode.readback.device_filter = true;
    decode.readback.run_ahead = 3;
    decode.readback.queued_steps = 2;
    let (_files, engine) = fixture_with_trace(decode, true)?;
    anyhow::ensure!(
        engine.encode("Hello world", false)? == OWNER_WARM_PROMPT,
        "original owner warm tokenizer population changed"
    );
    anyhow::ensure!(
        engine.core.model.guarded_embedding_layout()?.hidden == 128,
        "physical oracle requires tiny hidden128"
    );
    let metal = engine.core.device.as_metal_device()?;
    // Existing scoped pause: cold comparators do not train either route order.
    {
        let _paused = metal.pause_pass_plans()?;
        for prepared in [false, true] {
            anyhow::ensure!(
                !metal.pass_plans_enabled() && metal.arena().is_none(),
                "cold owner comparator unexpectedly planned"
            );
            owner_reused_request(&engine, prepared, true, false, "cold-owner")?;
        }
        anyhow::ensure!(
            !metal.pass_plans_enabled() && metal.arena().is_none(),
            "cold owner comparator installed plans"
        );
    }
    metal.enable_pass_plans(lloom_tensor::kernels::metal::arena::ArenaLayout::Packed);
    let (initial, _) = metal
        .pass_plan_report()
        .ok_or_else(|| anyhow::anyhow!("initial plans absent"))?;
    anyhow::ensure!(
        !initial
            .iter()
            .any(|(key, ..)| key.kind
                == lloom_tensor::kernels::metal::arena::PassKind::Decode { b: 1 }),
        "route order was already trained before owner warmup"
    );
    for (index, prepared) in [prepared_first, !prepared_first].into_iter().enumerate() {
        owner_reused_request(&engine, prepared, index == 0, true, "record-order")?;
    }
    let before = metal
        .pass_plan_counters()
        .ok_or_else(|| anyhow::anyhow!("before counters absent"))?;
    // Each warm request follows a cold request of the other route, as in the
    // record order. A warm request after a warm request is not this shape:
    // the owner replays the prefix (turn outcome Fresh, replayed 10).
    for (cold, warm) in [
        (prepared_first, !prepared_first),
        (!prepared_first, prepared_first),
    ] {
        owner_reused_request(&engine, cold, true, true, "served-order")?;
        owner_reused_request(&engine, warm, false, true, "served-order")?;
    }
    let after = metal
        .pass_plan_counters()
        .ok_or_else(|| anyhow::anyhow!("after counters absent"))?;
    anyhow::ensure!(
        after.served > before.served
            && after.recorded == before.recorded
            && after.evicted == before.evicted,
        "warm owner plans not stably served"
    );
    let (plans, _) = metal
        .pass_plan_report()
        .ok_or_else(|| anyhow::anyhow!("final plans absent"))?;
    for input in [
        lloom_tensor::backend::PassInput::TokenIds,
        lloom_tensor::backend::PassInput::PreparedEmbeddings,
    ] {
        anyhow::ensure!(
            plans.iter().any(|(key, covered, ..)| key.kind
                == lloom_tensor::kernels::metal::arena::PassKind::Decode { b: 1 }
                && key.input == input
                && *covered > 0),
            "owner warm route key absent: {input:?}"
        );
    }
    anyhow::ensure!(!metal.pass_open(), "owner warm pass left open");
    println!("OWNER_REUSED_COMPLETE prepared_first={prepared_first} owner=actual-engine native_sampler=true invalid_row_recovery=SEPARATE_GATE");
    Ok(())
}

#[test]
#[ignore = "explicit actual-owner warm native qualification; one selector/process; not timing"]
fn actual_owner_reused_prefix_prepared_then_token() -> Result<()> {
    owner_reused_order(true)
}

#[test]
#[ignore = "explicit actual-owner reverse warm native qualification; one selector/process; not timing"]
fn actual_owner_reused_prefix_token_then_prepared() -> Result<()> {
    owner_reused_order(false)
}

// Additive actual-owner stop/refusal controls; all earlier controls remain intact.
fn stop_owner_request() -> DecodeParams {
    DecodeParams {
        max_tokens: 16,
        top_k: Some(1),
        processors: lloom::generation::Processors {
            logit_bias: vec![(97, 100.)],
            ..Default::default()
        },
        ..params(ReadbackArm::CROWN, false, false)
    }
}

fn check_native_owner_stop_route(out: &Decoded, labels: &[(String, u64)]) {
    assert!(matches!(out.readback, ReadbackArm::Ring(_)));
    assert!(out
        .readback_note
        .as_deref()
        .is_some_and(|s| s.contains("native prepared")));
    assert!(
        labels
            .iter()
            .any(|(name, n)| name.contains("sampling_topk_mask") && *n > 0),
        "missing native sampler: {labels:?}"
    );
    assert!(
        labels
            .iter()
            .any(|(name, n)| name.contains("sampling_guarded_embedding") && *n > 0),
        "missing real guarded model input: {labels:?}"
    );
}

// Observe, do not repair, the actual owner after its public call has returned.
fn stopped_owner_state(engine: &NativeEngine, prompt: &[u32], out: &Decoded) -> (usize, usize) {
    let held = engine.single.lock();
    let held = held.as_ref().expect("public call returned owner custody");
    assert!(
        !held.needs_recovery,
        "ordinary successful stop is not failed-request recovery"
    );
    let seq = held.sequence.as_ref().expect("held sequence");
    assert_eq!(seq.tokens(), [prompt, out.tokens.as_slice()].concat());
    assert_eq!(seq.overrun(), out.stats.overrun);
    assert_eq!(seq.pos(), seq.tokens().len() + seq.overrun());
    (seq.tokens().len(), seq.overrun())
}

fn next_public_after_stop(
    engine: &NativeEngine,
    prompt: &[u32],
    prior: (usize, usize),
) -> Result<()> {
    // No reset, reconcile, fence or lease mutation here. Public preparation must
    // settle the previous overrun/rewind and disarm the previous forcing state.
    let request = DecodeParams {
        max_tokens: 4,
        ..stop_owner_request()
    };
    let out = run(engine, prompt, &request)?;
    assert_eq!(out.tokens, [97; 4]);
    assert_eq!(out.turn.boundary, prior.0);
    assert_eq!(out.turn.overrun, prior.1);
    assert!(out.thinking.is_none());
    assert!(!engine.single.lock().as_ref().unwrap().needs_recovery);
    println!(
        "OWNER_STOP_NEXT prior_boundary={} prior_overrun={} actual_turn={:?} tokens={:?}",
        prior.0, prior.1, out.turn, out.tokens
    );
    Ok(())
}

#[test]
#[ignore = "explicit native admission; one process per actual Metal API/contract"]
fn prepared_actual_eos_drains_then_next_public_request() -> Result<()> {
    let mut decode = crate::DecodeConfig::default();
    decode.readback.device_filter = true;
    decode.readback.run_ahead = 3;
    decode.readback.queued_steps = 2;
    let (_files, engine) = fixture(decode)?;
    let prompt = engine.encode("Hello world", false)?;
    let request = DecodeParams {
        eos_ids: vec![98],
        thinking: Some(lloom::generation::ThinkingBudget::new(
            0,
            vec![98],
            vec![97, 98],
        )?),
        ..stop_owner_request()
    };
    let census = engine
        .core
        .device
        .as_metal_device()?
        .dispatch_census()
        .open()
        .expect("isolated census");
    let mut delivered = String::new();
    let out = engine.complete_until(&prompt, &request, |s| {
        delivered.push_str(s);
        true
    })?;
    let labels = census.close();
    check_native_owner_stop_route(&out, &labels);
    assert_eq!(out.stop, Stop::Eos);
    assert_eq!(out.tokens, [97], "exclude EOS and all lookahead beyond it");
    assert_eq!(out.text, "a");
    assert_eq!(delivered, out.text, "no EOS/post-EOS fragment escapes");
    let prior = stopped_owner_state(&engine, &prompt, &out);
    println!("OWNER_ACTUAL_EOS stop={:?} delivered={delivered:?} tokens={:?} overrun={} labels={labels:?}", out.stop, out.tokens, out.stats.overrun);
    next_public_after_stop(&engine, &prompt, prior)
}

#[test]
#[ignore = "explicit native admission; one process per actual Metal API/contract"]
fn prepared_consumer_refusal_drains_then_next_public_request() -> Result<()> {
    let mut decode = crate::DecodeConfig::default();
    decode.readback.device_filter = true;
    decode.readback.run_ahead = 3;
    decode.readback.queued_steps = 2;
    let (_files, engine) = fixture(decode)?;
    let prompt = engine.encode("Hello world", false)?;
    let request = stop_owner_request();
    let census = engine
        .core
        .device
        .as_metal_device()?
        .dispatch_census()
        .open()
        .expect("isolated census");
    let mut offered = Vec::new();
    let out = engine.complete_until(&prompt, &request, |s| {
        offered.push(s.to_owned());
        false
    })?;
    let labels = census.close();
    check_native_owner_stop_route(&out, &labels);
    assert_eq!(out.stop, Stop::Cancelled);
    assert_eq!(
        offered.len(),
        1,
        "refused consumer must not receive later fragments"
    );
    assert!(!offered[0].is_empty());
    assert!(offered[0].bytes().all(|b| b == b'a'));
    // Existing TextStream contract: false rejects that fragment; generated ids
    // remain separate from delivered text, so no zero-generated assertion.
    assert!(
        out.text.is_empty(),
        "first refused fragment is not delivered text"
    );
    assert!(!out.tokens.is_empty());
    assert!(out.tokens.len() <= request.max_tokens);
    assert!(out.tokens.iter().all(|&id| id == 97));
    let prior = stopped_owner_state(&engine, &prompt, &out);
    println!("OWNER_CONSUMER_REFUSAL offered={offered:?} delivered={:?} generated={:?} overrun={} labels={labels:?}", out.text, out.tokens, out.stats.overrun);
    next_public_after_stop(&engine, &prompt, prior)
}

// Fault only at real Metal capability/preparation boundaries. Body delegates to
// the loaded model and counts any accidental entry; no alternative decode loop.
struct RefusePreparedOwner<'a> {
    inner: &'a Inner,
    preparation: bool,
    layout_calls: AtomicUsize,
    preparation_calls: AtomicUsize,
    body_calls: AtomicUsize,
}
impl ModelTrait for RefusePreparedOwner<'_> {
    type Config = Config;
    fn from_weights(_: VarBuilder, _: &Config) -> lloom_tensor::Result<Self> {
        lloom_tensor::bail!("borrowed actual owner capability fixture")
    }
}
impl CausalLM for RefusePreparedOwner<'_> {
    type Cache = <Inner as CausalLM>::Cache;
    fn new_cache(&self, cfg: &Config, d: &Device) -> lloom_tensor::Result<Self::Cache> {
        self.inner.new_cache(cfg, d)
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
        self.body_calls.fetch_add(1, Ordering::Relaxed);
        self.inner.forward(ids, pos, cache)
    }
    fn forward_embeds(
        &self,
        x: &Tensor,
        pos: usize,
        cache: &mut Self::Cache,
    ) -> lloom_tensor::Result<Tensor> {
        self.body_calls.fetch_add(1, Ordering::Relaxed);
        self.inner.forward_embeds(x, pos, cache)
    }
    fn guarded_embedding_layout(
        &self,
    ) -> lloom_tensor::Result<lloom_tensor::quantized::GuardedEmbeddingLayout> {
        self.layout_calls.fetch_add(1, Ordering::Relaxed);
        if !self.preparation {
            lloom_tensor::bail!("OWNER_NATIVE_CAPABILITY_REFUSAL");
        }
        self.inner.guarded_embedding_layout()
    }
    fn check_guarded_embedding(&self, out: &Tensor) -> lloom_tensor::Result<()> {
        self.preparation_calls.fetch_add(1, Ordering::Relaxed);
        self.inner.check_guarded_embedding(out)?;
        lloom_tensor::bail!("OWNER_NATIVE_PREPARATION_REFUSAL")
    }
    fn prepare_pass(&self, cache: &mut Self::Cache, n: usize) -> lloom_tensor::Result<()> {
        self.inner.prepare_pass(cache, n)
    }
    fn checkpoint_at(&self, cache: &mut Self::Cache, pos: usize) -> lloom_tensor::Result<()> {
        self.inner.checkpoint_at(cache, pos)
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
    fn eos_gate_cores(&self) -> bool {
        self.inner.eos_gate_cores()
    }
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
}

#[test]
#[ignore = "explicit native admission; one process per actual Metal API/contract"]
fn prepared_metal_capability_and_preparation_refuse_before_first_token() -> Result<()> {
    let mut decode = crate::DecodeConfig::default();
    decode.readback.device_filter = true;
    decode.readback.run_ahead = 3;
    decode.readback.queued_steps = 2;
    let (_files, engine) = fixture(decode)?;
    assert!(engine.core.device.is_metal());
    let prompt = engine.encode("Hello world", false)?;
    for preparation in [false, true] {
        let fault = RefusePreparedOwner {
            inner: &engine.core.model,
            preparation,
            layout_calls: AtomicUsize::new(0),
            preparation_calls: AtomicUsize::new(0),
            body_calls: AtomicUsize::new(0),
        };
        let mut fragments = 0;
        let error = {
            let mut lease = engine.checkout_single()?;
            let config = lease.extra().decode.clone();
            lease
                .with_model(&fault, &engine.core.device, config, |session, _| {
                    session.decode_until(&prompt, engine.detok(), &stop_owner_request(), |_| {
                        fragments += 1;
                        true
                    })
                })
                .expect_err("actual Metal capability/preparation must refuse")
        };
        let expected = if preparation {
            "OWNER_NATIVE_PREPARATION_REFUSAL"
        } else {
            "OWNER_NATIVE_CAPABILITY_REFUSAL"
        };
        assert_eq!(error.to_string(), expected);
        assert_eq!(fault.layout_calls.load(Ordering::Relaxed), 1);
        assert_eq!(
            fault.preparation_calls.load(Ordering::Relaxed),
            usize::from(preparation)
        );
        assert_eq!(fault.body_calls.load(Ordering::Relaxed), 0);
        assert_eq!(fragments, 0);
        assert!(engine.single.lock().as_ref().unwrap().needs_recovery);
        // The real owner performs its own recovery, including any allocated
        // preparation bytes. No manual fence/reset/marker clear intervenes.
        let next = run(
            &engine,
            &prompt,
            &DecodeParams {
                max_tokens: 4,
                ..stop_owner_request()
            },
        )?;
        assert_eq!(next.tokens, [97; 4]);
        assert_eq!(next.reused, 0);
        assert!(!engine.single.lock().as_ref().unwrap().needs_recovery);
        println!("OWNER_EARLY_REFUSAL preparation={preparation} error={error} fragments={fragments} bodies=0 next_tokens={:?}", next.tokens);
    }
    Ok(())
}
