//! Ignored bounded native caller proposal. Source compilation is not execution
//! authorization, complete admission, checkpoint parity or HTTP socket evidence.
use super::dispatch_systemone;
use crate::app::causal_loaded::served::{self, Files, Predictor, Request};
use crate::app::causal_loaded::CausalPolicySource;
use crate::app::served::Served;
use crate::decision::{
    Answer, CausalLabels, CausalRequest, CausalTask, NoulAliases, OrderedJson, Question,
    StateNullability, SystemOneTemplate, SystemOneTemplateKind,
};
use crate::{DecodeConfig, DecodeParams, ReadbackArm};
use lloom::nn::memory::{pool_bound_bytes, MachineLoad};
use lloom::nn::CausalLM;
use lloom_tensor::machine::{Gate, GateConfig, Peak};
use lloom_tensor::memory_admission::{configure_process_account, process_account};
use lloom_tensor::quantized::gguf_file;
use lloom_tensor::{Device, Result as TensorResult, Tensor};
use sha2::{Digest, Sha256};
use std::io::{Cursor, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokenizers::Tokenizer;

#[path = "../../../lloom/src/models/quantized_lfm2/fixture.rs"]
mod fixture;

const MIB: usize = 1024 * 1024;
const PROCESS_CAP: usize = 256 * MIB;
const MODEL_DEMAND: usize = 32 * MIB;
const MODEL_QUOTA: usize = 64 * MIB;
const REQUEST_DEMAND: usize = 16 * MIB;
const FREE_POOL_EACH: usize = MIB;
const NAMED: &str = include_str!("../decision/fixtures/d1-systemone.jinja");

#[derive(Default)]
struct Calls {
    generation: AtomicUsize,
    prepared: AtomicUsize,
    raw: AtomicUsize,
}
struct FixedPredictor(Arc<Calls>);
impl Predictor for FixedPredictor {
    fn predicted_extra(&self, request: Request<'_>) -> TensorResult<usize> {
        match request {
            Request::Generation { prompt, params } => {
                assert!(prompt.len() <= 3 && params.max_tokens == 1);
                assert!(matches!(params.readback, ReadbackArm::Blocking));
                self.0.generation.fetch_add(1, Ordering::Relaxed);
            }
            Request::PreparedDecisions(request) => {
                assert_eq!(request.max_prompt_tokens(), 3);
                assert_eq!(request.input_tokens(), 6);
                self.0.prepared.fetch_add(1, Ordering::Relaxed);
            }
            Request::Decisions(_) => {
                self.0.raw.fetch_add(1, Ordering::Relaxed);
            }
            _ => lloom_tensor::bail!("this bounded fixture did not admit another producer"),
        }
        // Fixed caller-supplied overcoverage, NOT complete physical/host pricing.
        Ok(REQUEST_DEMAND)
    }
}

struct FilesOwned(PathBuf);
impl FilesOwned {
    fn new() -> anyhow::Result<Self> {
        let path = std::env::temp_dir().join(format!("lloom947-loaded-{}", std::process::id()));
        std::fs::create_dir(&path)?; // Refuse an existing directory; never overwrite it.
        Ok(Self(path))
    }
    fn put(&self, name: &str, bytes: &[u8]) -> anyhow::Result<PathBuf> {
        let path = self.0.join(name);
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        Ok(path)
    }
}
impl Drop for FilesOwned {
    fn drop(&mut self) {
        // Only these fresh owned fixture files; no recursive directory removal.
        for name in ["fixture.gguf", "tokenizer.json"] {
            let _ = std::fs::remove_file(self.0.join(name));
        }
        let _ = std::fs::remove_dir(&self.0);
    }
}
fn tokenizer() -> anyhow::Result<Tokenizer> {
    let model = tokenizers::models::wordlevel::WordLevel::builder()
        .vocab(
            ["A", "B", "a", "b", "c", "d", "e", "[UNK]"]
                .into_iter()
                .enumerate()
                .map(|(id, name)| (name.into(), id as u32))
                .collect(),
        )
        .unk_token("[UNK]".into())
        .build()
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(
        tokenizers::pre_tokenizers::whitespace::WhitespaceSplit,
    ));
    Ok(tokenizer)
}
fn choice() -> Question<OrderedJson> {
    Question::Choice {
        instructions: OrderedJson::String("choose".into()),
        criteria: vec![("A".into(), None), ("B".into(), None)],
    }
}
fn prepared() -> TensorResult<CausalRequest> {
    let tasks = [vec![2, 3, 4], vec![2, 3, 5]]
        .into_iter()
        .enumerate()
        .map(|(i, prompt)| {
            let labels = CausalLabels::systemone(&choice(), |code| match code.trim() {
                "A" => Ok(vec![0]),
                "B" => Ok(vec![1]),
                _ => lloom_tensor::bail!("bounded caller supplied an unexpected label"),
            })?;
            CausalTask::new(
                format!("q{i}"),
                choice(),
                prompt,
                labels.into_readout(),
                1.0.try_into()?,
            )
        })
        .collect::<TensorResult<Vec<_>>>()?;
    // Complete explicit state-token prefix, not a guessed prompt LCP.
    CausalRequest::new(tasks, &[2, 3], 3)
}
fn cpu_replay(
    model: &lloom::models::quantized_lfm2::Model,
    prompts: &[Vec<u32>],
) -> TensorResult<Vec<[f64; 2]>> {
    let mut result = Vec::with_capacity(prompts.len());
    for prompt in prompts {
        assert!(!prompt.is_empty() && prompt.len() <= 32);
        let mut cache = model.new_cache()?;
        let mut logits = None;
        for (pos, &id) in prompt.iter().enumerate() {
            logits = Some(CausalLM::forward(
                model,
                &Tensor::new(&[[id]], &Device::Cpu)?,
                pos,
                &mut cache,
            )?);
        }
        let logits = logits
            .expect("bounded nonempty prompt")
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(logits.len(), 8);
        let a = f64::from(logits[0]);
        let b = f64::from(logits[1]);
        let max = a.max(b);
        let a = (a - max).exp();
        let b = (b - max).exp();
        result.push([a / (a + b), b / (a + b)]);
    }
    Ok(result)
}
fn check_probability(actual: f64, expected: f64) {
    assert!(actual.is_finite() && expected.is_finite());
    assert!(
        (actual - expected).abs() < 3e-5,
        "loaded posterior {actual} != replay {expected}"
    );
}

fn exercise(device: &Device) -> anyhow::Result<()> {
    let bytes = fixture::fixture(Some(NAMED))?;
    let files = FilesOwned::new()?;
    let gguf = files.put("fixture.gguf", &bytes)?;
    let tokenizer = tokenizer()?;
    let tokenizer_path = files.put(
        "tokenizer.json",
        tokenizer
            .to_string(false)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
            .as_bytes(),
    )?;
    eprintln!(
        "BOUND_SYNTHETIC_GGUF_BYTES={} SHA256={:x}",
        bytes.len(),
        Sha256::digest(&bytes)
    );

    let machine = MachineLoad {
        gate: Gate::new(GateConfig {
            off: true,
            ..Default::default()
        }),
        label: "bounded-real-loaded-fixture".into(),
        rows: 1,
        kv_tokens: 32,
        prefill_tile: 1,
    };
    assert_eq!(pool_bound_bytes(device), 2 * FREE_POOL_EACH);
    let quote = lloom_tensor::metal_backend::MetalAllocationDomain::requested_metadata_bytes()?;
    assert!(quote <= 16 * MIB, "reviewed quote bound exceeded");
    let load = machine.admit_context(
        device,
        Peak::new().with(
            "bounded synthetic model/caller envelope",
            MODEL_DEMAND as u64,
        ),
        Some(MODEL_QUOTA),
    )?;
    let domain = load.domain().clone();
    assert_eq!(
        domain.snapshot().pending,
        MODEL_DEMAND + 2 * FREE_POOL_EACH + quote
    );
    let calls = Arc::new(Calls::default());
    let mut decode = DecodeConfig::default();
    crate::options::ReadbackConfig::apply_spelling(&mut decode.readback, "blocking")
        .map_err(anyhow::Error::msg)?;
    decode.prefill.tile = 1;
    let engine = served::load_lfm2_gguf(
        Files {
            checkpoint: &gguf,
            tokenizer: &tokenizer_path,
        },
        Some(CausalPolicySource::Named {
            temperature: 1.0.try_into()?,
            max_options: 2,
            noul_aliases: NoulAliases::TrueFalseOnly,
            state_nullability: StateNullability::AllowNull,
        }),
        device,
        load,
        decode,
        served::Config {
            model_name: "synthetic21-f32".into(),
            bos_token: String::new(),
            bos_id: None,
            eos_ids: vec![7],
            seed_text: "a".into(),
            chat_template: None,
        },
        FixedPredictor(Arc::clone(&calls)),
    )?;
    assert_eq!(domain.snapshot().pending, 0, "load demand remained open");
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
        || -> anyhow::Result<()> {
            // One separate tiny CPU oracle, not a replacement production owner.
            let mut reader = Cursor::new(&bytes);
            let oracle = lloom::models::quantized_lfm2::Model::from_gguf(
                gguf_file::Content::read(&mut reader)?,
                &mut reader,
                &Device::Cpu,
            )?;
            let params = DecodeParams {
                max_tokens: 1,
                readback: ReadbackArm::Blocking,
                prefill_tile: 1,
                ..Default::default()
            };
            let first = engine.complete_admitted(&[2, 3], &params, |_| true)?;
            assert_eq!(first.tokens.len(), 1);
            let decision = engine.decide_prepared(&prepared()?)?;
            let expected = cpu_replay(&oracle, &[vec![2, 3, 4], vec![2, 3, 5]])?;
            assert_eq!(decision.input_tokens(), 6);
            assert_eq!(decision.answers().len(), 2);
            for (i, (id, answer)) in decision.answers().iter().enumerate() {
                assert_eq!(id, &format!("q{i}"));
                let Answer::Choice { probabilities, .. } = answer else {
                    panic!("actual prepared choice was not returned");
                };
                assert_eq!(probabilities.len(), 2);
                for (j, (label, probability)) in probabilities.iter().enumerate() {
                    assert_eq!(label, if j == 0 { "A" } else { "B" });
                    check_probability(*probability, expected[i][j]);
                }
            }

            let template = SystemOneTemplate::compile(NAMED, SystemOneTemplateKind::D1Causal)?;
            let state = OrderedJson::String("a".into());
            let mut prompts = Vec::new();
            for id in ["z", "a"] {
                let text = template.render(
                    id,
                    &state,
                    &choice(),
                    &["A".into(), "B".into()],
                    crate::decision::SystemOneMedia::Text,
                )?;
                let ids = tokenizer
                    .encode(text, false)
                    .map_err(|e| anyhow::anyhow!(e.to_string()))?
                    .get_ids()
                    .to_vec();
                assert!(ids.len() <= 32 && ids.iter().all(|&id| id < 8));
                prompts.push(ids);
            }
            let raw_expected = cpu_replay(&oracle, &prompts)?;
            let input_count: usize = prompts.iter().map(Vec::len).sum();
            let text = dispatch_systemone(
        &engine,
        "/v1/systemone",
        br#"{"state":"a","questions":{"z":{"type":"choice","instructions":"choose","criteria":{"A":null,"B":null}},"a":{"type":"choice","instructions":"choose","criteria":{"A":null,"B":null}}}}"#,
    ).expect("actual endpoint route")?;
            assert!(text.starts_with(r#"{"answers":{"z":{"type":"choice""#));
            let response: serde_json::Value = serde_json::from_str(&text)?;
            assert_eq!(response["usage"]["input_tokens"], input_count);
            assert_eq!(response["usage"]["output_tokens"], 0);
            for (i, id) in ["z", "a"].iter().enumerate() {
                for (j, label) in ["A", "B"].iter().enumerate() {
                    check_probability(
                        response["answers"][id]["probabilities"][label]
                            .as_f64()
                            .expect("actual numeric posterior"),
                        raw_expected[i][j],
                    );
                }
            }
            let second = Served::complete_until(&engine, &[2, 3], &params, &mut |_| true)?;
            assert_eq!(
                second.tokens, first.tokens,
                "decisions changed held ordinary generation"
            );
            assert_eq!(calls.generation.load(Ordering::Relaxed), 2);
            assert_eq!(calls.prepared.load(Ordering::Relaxed), 1);
            assert_eq!(calls.raw.load(Ordering::Relaxed), 1);
            assert_eq!(domain.snapshot().pending, 0);
            eprintln!("REAL_LOADED_ORDINARY_PREPARED_RAW_ORDINARY_OK");
            Ok(())
        },
    ));
    // The owner is still held on request Err/panic. Attempt prefix DATA closure
    // before its drop, outside callback unwinding; refusal is not safe teardown.
    let terminal = (|| -> TensorResult<()> {
        let metal = device.as_metal_device()?;
        let marker = metal.flush_marker()?;
        metal.wait_marker_prefix_data(&marker)
    })();
    eprintln!("BOUNDED_HELD_OWNER_PREFIX_RESULT={terminal:?}");
    terminal?;
    drop(engine);
    match outcome {
        Ok(result) => result,
        Err(payload) => std::panic::resume_unwind(payload),
    }
}

#[test]
#[ignore = "source proposal: exact native runtime requires independent bounded review"]
fn actual_typed_loaded_owner_serves_generation_and_ordered_decisions() -> anyhow::Result<()> {
    eprintln!("BOUNDED_REAL_LOADED_TEST_START");
    let device = Device::new_metal(0)?;
    eprintln!(
        "BOUNDED_COMMAND_BUFFER_KIND={:?}",
        device.as_metal_device()?.command_buffer_kind()
    );
    device.as_metal_device()?.set_free_budget(FREE_POOL_EACH);
    // This isolated exact test process deliberately exercises the user window.
    configure_process_account(Some(PROCESS_CAP))?;
    assert_eq!(process_account().cap(), Some(PROCESS_CAP));
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| exercise(&device)));
    // Also attempt the real associated prefix data barrier on Err/panic, outside
    // callback unwinding. Preserve refusals; never report them as safe teardown.
    let terminal = (|| -> TensorResult<()> {
        let metal = device.as_metal_device()?;
        let marker = metal.flush_marker()?;
        metal.wait_marker_prefix_data(&marker)
    })();
    eprintln!("BOUNDED_FINAL_PREFIX_RESULT={terminal:?}");
    terminal?;
    match outcome {
        Ok(result) => result,
        Err(payload) => std::panic::resume_unwind(payload),
    }
}
