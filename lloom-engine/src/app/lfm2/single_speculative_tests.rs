//! Actual single-sequence Engine handlers over the existing CPU tiny pair.
//! UNCOMPILED/UNRUN. No loader, native retirement or complete admission proof.

use super::*;
use lloom::generation::{Processors, ThinkingBudget, ThinkingReport};
use lloom::models::lfm2_testing::Lfm2Tiny;
use lloom::nn::{CacheRewind, SeqId, Speculator};
use lloom::profile::RewindableExact;
use lloom::testing::fixture::Fixture;

type TinyEngine = Engine<RewindableExact>;

fn engine_fixture() -> Result<(Fixture<Lfm2Tiny>, TinyEngine)> {
    let (fixture, model, config, drafter) = super::speculative_schedule_tests::pair()?;
    let device = Device::Cpu;
    let cache = State::<RewindableExact>::new_with(true, DType::F32, &config, &device, 128)?;
    let mut spec = DSparkSpeculator::new_with_ctx(
        drafter,
        model.draft_embedding().clone(),
        model.lm_head_linear().clone(),
        1,
        CtxKv::Contiguous,
    )?;
    // Match the actual loader option, not only the Engine report metadata.
    spec.set_max_draft(2);
    assert_eq!(spec.max_draft(), 2);
    let tokenizer = Tokenizer::new(
        tokenizers::models::wordlevel::WordLevel::builder()
            .vocab((0..48u32).map(|id| (format!("t{id:02}"), id)).collect())
            .unk_token("t47".into())
            .build()
            .map_err(lloom_tensor::Error::wrap)?,
    );
    // No files are mapped for this unused load-report metadata. The actual
    // model and drafter above come from the shared in-memory fixture builder.
    // SAFETY: the empty path set maps no file that could be modified.
    let (_, lm_record) = unsafe { RecordingBackend::over_safetensors(&[] as &[PathBuf])? };
    assert_eq!(lm_record.totals(), (0, 0));
    let mut decode = crate::DecodeConfig::default();
    decode.prefill.tile = 16;
    Ok((
        fixture,
        Engine {
            core: Arc::new(Core { model, device }),
            text: Arc::new(Text {
                tokenizer,
                template: None,
                bos_token: String::new(),
                bos_id: None,
                eos_ids: NEVER_EOS.to_vec(),
                seed_tokens: vec![2, 3],
                think: Default::default(),
                gen_prompts: Default::default(),
            }),
            model_name: "cpu-tiny-handler".into(),
            kv_capacity: 128,
            readback: ReadbackArm::Blocking,
            axes_line: config.axes_line(),
            model: Default::default(),
            generation: None,
            executor: Default::default(),
            reuse_prefix: true,
            batch: 1,
            dspark_k: Some(2),
            prefill_tile: 16,
            prefill_chunk: 0,
            prefill_batch: PrefillBatch::one(),
            prefill_decode: lloom::nn::PrefillDecode::Ride,
            memory_fraction: 0.0,
            mem_sample: false,
            rows: RowsArm::Private,
            pool: None,
            share_prefix: false,
            lm_record,
            vision: None,
            last_vision: Mutex::new(Value::Null),
            config,
            dtype: DType::F32,
            dspark_setup: None,
            installed: Default::default(),
            warm_in_flight: None,
            load_warnings: Vec::new(),
            machine: None,
            private_cache: PrivateCachePolicy {
                capacity: 128,
                bucket: 128,
                grow_fraction: None,
            },
            pin_refused: None,
            decisions: None,
            decision_ragged: false,
            trunks: Default::default(),
            single: parking_lot::Mutex::new(Some(RequestState {
                sequence: Some(Sequence::new(cache)),
                rings: Some(RingPool::new()),
                extra: SingleExtra {
                    spec: Some(spec),
                    decode,
                    image_keys: Vec::new(),
                },
                needs_recovery: false,
            })),
            sched: None,
            sched_thread: Mutex::new(None),
        },
    ))
}

fn prompt() -> Vec<u32> {
    (0..37).map(|i| i % 47).collect()
}

fn params(cap: usize) -> DecodeParams {
    DecodeParams {
        max_tokens: cap,
        eos_ids: NEVER_EOS.to_vec(),
        readback: ReadbackArm::Blocking,
        turn_tail: 1,
        ..Default::default()
    }
}

fn sampled(cap: usize) -> DecodeParams {
    DecodeParams {
        seed: 7,
        temperature: Some(0.7),
        top_k: Some(1),
        processors: Processors {
            logit_bias: vec![(0, 100.0)],
            frequency_penalty: 1.0,
            ..Default::default()
        },
        logits_digest: true,
        ..params(cap)
    }
}

fn published_ids(text: &str) -> Vec<u32> {
    // Each vocabulary spelling is self-delimiting even when separate fragments
    // meet without a space. No special token or partial character is present.
    text.split('t')
        .skip(1)
        .map(|id| id.trim().parse().unwrap())
        .collect()
}

fn complete(engine: &TinyEngine, prompt: &[u32], params: &DecodeParams) -> Result<Decoded> {
    assert_eq!(engine.batch, 1);
    assert!(engine.sched.is_none());
    assert!(lock(&engine.sched_thread).is_none());
    let mut published = String::new();
    // Use the Served forwarding edge as well as the concrete Engine handler.
    let served: &dyn super::super::served::Served = engine;
    let out = served.complete_until(prompt, params, &mut |part| {
        published.push_str(part);
        true
    })?;
    assert_eq!(published, out.text);
    assert_eq!(published_ids(&published), out.tokens);
    assert_eq!(out.readback, ReadbackArm::Speculative { k: 2 });
    assert_eq!(out.stats.produced, out.tokens.len());
    assert_eq!(out.stats.overrun, 0);
    Ok(out)
}

#[derive(Debug, PartialEq, Eq)]
struct Held {
    recovery: bool,
    ids: Vec<u32>,
    pos: usize,
    kv: usize,
    overrun: usize,
    phase: usize,
    checkpoint: Option<usize>,
    context: Option<usize>,
    keyed: Option<(usize, usize)>,
    resets: u64,
}

fn held(engine: &TinyEngine, checkpoint: usize) -> Held {
    let slot = engine.single.lock();
    let state = slot.as_ref().expect("handler lost request custody");
    assert!(state.rings.is_some());
    let seq = state.sequence.as_ref().unwrap();
    let spec = state.extra.spec.as_ref().expect("handler lost drafter");
    Held {
        recovery: state.needs_recovery,
        ids: seq.tokens().to_vec(),
        pos: seq.pos(),
        kv: seq.cache().kv_len(),
        overrun: seq.overrun(),
        phase: seq.cache().conv_phase(),
        checkpoint: seq.cache().restore_point(checkpoint),
        context: spec.context_len(SeqId(0)),
        keyed: spec.ctx_state(SeqId(0)),
        resets: spec.ctx_resets(),
    }
}

fn exact_state(engine: &TinyEngine, full: &[u32], pos: usize) -> Held {
    let state = held(engine, 36);
    assert!(!state.recovery);
    assert_eq!(state.ids, full[..pos]);
    assert_eq!(state.pos, pos);
    assert_eq!(state.kv, pos);
    assert_eq!(state.overrun, 0);
    assert_eq!(state.context, Some(pos));
    assert_eq!(state.keyed, Some((pos, pos)));
    state
}

#[test]
fn actual_b1_greedy_handler_publishes_caps_and_runs_verify() -> Result<()> {
    let (_fixture, engine) = engine_fixture()?;
    let prompt = prompt();
    for cap in [1, 4] {
        let mut request = params(cap);
        request.cold = true;
        let out = complete(&engine, &prompt, &request)?;
        assert_eq!(out.stop, Stop::Length);
        assert_eq!(out.tokens.len(), cap);
        assert_eq!(out.stats.prefill_units, 3);
        assert_eq!(out.prefilled, 37);
        assert!(out.logits_sha256.is_none());
        let full = [prompt.as_slice(), &out.tokens].concat();
        if cap == 1 {
            assert!(out.stats.rounds.is_empty());
            let state = exact_state(&engine, &full, 37);
            assert_eq!(state.checkpoint, Some(36));
        } else {
            assert!(!out.stats.rounds.is_empty(), "no actual verify round ran");
            let pos = held(&engine, 36).pos;
            assert!((40..=41).contains(&pos), "cap4 input frontier {pos}");
            exact_state(&engine, &full, pos);
        }
    }
    Ok(())
}

#[test]
fn actual_b1_sampled_support_positive_then_logprobs_refusal_and_next_request() -> Result<()> {
    for cap in [3, 2] {
        let (_fixture, engine) = engine_fixture()?;
        let prompt = prompt();
        let mut request = sampled(cap);
        request.thinking = Some(ThinkingBudget::new(2, vec![1], vec![1])?);
        request.eos_ids = vec![1];
        let first = complete(&engine, &prompt, &request)?;
        assert_eq!(first.tokens, [0, 0]);
        let expected_stop = if cap == 3 { Stop::Eos } else { Stop::Length };
        assert_eq!(first.stop, expected_stop);
        assert!(first.logits_sha256.is_some());
        assert_eq!(
            first.thinking,
            Some(ThinkingReport {
                thought: 2,
                fired: false,
                closed_by: None,
            })
        );
        let full = [prompt.as_slice(), &[0, 0]].concat();
        let pos = held(&engine, 36).pos;
        if cap == 3 {
            assert_eq!(pos, 39);
        } else {
            assert!((38..=39).contains(&pos), "cap2 input frontier {pos}");
        }
        let before = exact_state(&engine, &full, pos);
        let mut refused = request.clone();
        refused.logprobs = Some(crate::decode::LogprobRequest {
            top: 1,
            rows: false,
        });
        let mut callbacks = 0;
        let error = engine
            .complete_until(&prompt, &refused, |_| {
                callbacks += 1;
                true
            })
            .unwrap_err();
        assert!(
            error.to_string().contains("per-token logprobs"),
            "{error:#}"
        );
        assert_eq!(callbacks, 0);
        let after = held(&engine, 36);
        assert_eq!(
            after,
            Held {
                recovery: true,
                ..before
            }
        );
        let next = complete(&engine, &prompt, &request)?;
        assert_eq!(next.tokens, [0, 0]);
        assert_eq!(next.stop, expected_stop);
        assert_eq!(next.reuse, ReuseAction::Fresh);
        assert_eq!(next.reused, 0);
        assert_eq!(next.prefilled, 37);
        assert_eq!(
            next.turn.boundary, 0,
            "deferred recovery did not reset before prepare"
        );
        assert_eq!(next.logits_sha256, first.logits_sha256);
        exact_state(&engine, &full, pos);
    }
    Ok(())
}

#[test]
fn actual_b1_capacity_refusal_keeps_prefix_for_next_request() -> Result<()> {
    let (_fixture, engine) = engine_fixture()?;
    let prompt = prompt();
    let first = complete(&engine, &prompt, &sampled(1))?;
    assert_eq!(first.tokens, [0]);
    let before = exact_state(&engine, &prompt, 37);
    let mut callbacks = 0;
    let error = engine
        .complete_until(&prompt, &sampled(128), |_| {
            callbacks += 1;
            true
        })
        .unwrap_err();
    assert!(
        error.to_string().contains("exceeds the KV capacity 128"),
        "{error:#}"
    );
    assert_eq!(callbacks, 0);
    assert_eq!(
        held(&engine, 36),
        before,
        "pre-checkout refusal moved held state"
    );
    let full = [prompt.as_slice(), &[0, 3, 4]].concat();
    let next = complete(&engine, &full, &sampled(1))?;
    assert_eq!(next.tokens, [0]);
    assert_eq!(next.stop, Stop::Length);
    assert_eq!(next.reuse, ReuseAction::Extend);
    assert_eq!(next.reused, 37);
    assert_eq!(next.prefilled, 3);
    exact_state(&engine, &full, 40);
    Ok(())
}

#[test]
fn actual_b1_callback_cancel_and_stop_string_continue_without_error_recovery() -> Result<()> {
    for cancel in [true, false] {
        let (_fixture, engine) = engine_fixture()?;
        let prompt = prompt();
        let mut request = sampled(4);
        if !cancel {
            request.stop_strings = vec!["t00".into()];
        }
        let mut offered = String::new();
        let mut callbacks = 0;
        let out = engine.complete_until(&prompt, &request, |part| {
            callbacks += 1;
            offered.push_str(part);
            false
        })?;
        assert_eq!(out.readback, ReadbackArm::Speculative { k: 2 });
        assert_eq!(
            out.stop,
            if cancel {
                Stop::Cancelled
            } else {
                Stop::StopString
            }
        );
        assert_eq!(
            out.tokens,
            [0],
            "generated ids are not the delivered text frontier"
        );
        assert_eq!(out.stats.produced, 1);
        assert!(out.stats.rounds.is_empty());
        assert_eq!(callbacks, usize::from(cancel));
        assert_eq!(offered, if cancel { "t00" } else { "" });
        assert_eq!(
            out.text, "",
            "neither a refused callback offer nor a stop-string match is delivered"
        );
        exact_state(&engine, &prompt, 37);
        let full = [prompt.as_slice(), &[3, 4]].concat();
        let next = complete(&engine, &full, &sampled(1))?;
        assert_eq!(next.tokens, [0]);
        assert_eq!(next.stop, Stop::Length);
        assert_eq!(next.reuse, ReuseAction::Extend);
        assert_eq!(next.reused, 37);
        assert_eq!(next.prefilled, 2);
        exact_state(&engine, &full, 39);
    }
    Ok(())
}

#[test]
fn actual_b1_callback_unwind_after_verify_then_next_request_is_fresh() -> Result<()> {
    let (_fixture, engine) = engine_fixture()?;
    let prompt = prompt();
    let mut callbacks = 0;
    let mut delivered = String::new();
    let failure = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        engine.complete_until(&prompt, &sampled(4), |part| {
            callbacks += 1;
            if callbacks == 2 {
                panic!("bounded B1 callback unwind after verify");
            }
            delivered.push_str(part);
            true
        })
    }));
    let payload = failure.expect_err("actual handler did not reach second callback");
    assert_eq!(
        payload.downcast_ref::<&str>().copied(),
        Some("bounded B1 callback unwind after verify")
    );
    assert_eq!(callbacks, 2);
    assert_eq!(published_ids(&delivered), [0]);
    let failed = held(&engine, 36);
    assert!(failed.recovery);
    assert_eq!(
        failed.ids, prompt,
        "error path did not finish sequence bookkeeping"
    );
    assert!(
        (38..=40).contains(&failed.pos),
        "no verified input advanced"
    );
    assert_eq!(failed.kv, failed.pos);
    assert_eq!(failed.context, Some(failed.pos));
    assert_eq!(failed.keyed, Some((failed.pos, failed.pos)));
    assert_eq!(
        failed.checkpoint,
        Some(36),
        "Drop ran cache recovery during unwind"
    );
    let full = [prompt.as_slice(), &[0, 3, 4]].concat();
    let mut next_request = sampled(4);
    next_request.seed = 1234;
    next_request.top_k = Some(7);
    next_request.processors.logit_bias.clear();
    let next = complete(&engine, &full, &next_request)?;
    assert_eq!(next.tokens.len(), 4);
    assert_eq!(next.stop, Stop::Length);
    assert_eq!(next.reuse, ReuseAction::Fresh);
    assert_eq!(next.turn.boundary, 0);
    assert_eq!(next.reused, 0);
    assert_eq!(next.prefilled, 40);
    let (_reference_fixture, reference) = engine_fixture()?;
    let fresh = complete(&reference, &full, &next_request)?;
    assert_eq!(next.tokens, fresh.tokens);
    assert_eq!(next.text, fresh.text);
    assert_eq!(next.logits_sha256, fresh.logits_sha256);
    assert_eq!(held(&engine, 39), held(&reference, 39));
    Ok(())
}
