//! Actual shared raw-policy preparation/certificate controls only. No Loaded,
//! model, tensors, native device, sockets or forward; not full caller acceptance.
use super::*;
use std::cell::RefCell;

const SOURCE: &str = include_str!("../../decision/fixtures/d1-systemone.jinja");
fn policy(
    source: &str,
    max_options: usize,
    state_nullability: StateNullability,
) -> Result<CausalPolicy> {
    Ok(CausalPolicy {
        template: SystemOneTemplate::compile(source, SystemOneTemplateKind::D1Causal)?,
        calibration: crate::decision::binding::Calibration::uniform(Temperature::try_from(2.5)?),
        max_options,
        noul_aliases: NoulAliases::TrueFalseOnly,
        state_nullability,
    })
}
fn body(text: &str) -> OrderedJson {
    assert!(text.len() < 1024);
    OrderedJson::from_slice(text.as_bytes()).unwrap()
}

#[test]
fn certified_preparation_preserves_ordered_whole_prompts_and_usage() -> Result<()> {
    let policy = policy(SOURCE, 2, StateNullability::RequireValue)?;
    assert!(policy.template.has_causal_certificate());
    let body = body(
        r#"{"state":"shared","questions":{"z":{"type":"choice","instructions":"first","criteria":{"B":"bee","A":"aye"}},"a":{"type":"choice","instructions":"second","criteria":{"A":"aye","B":"bee"}}}}"#,
    );
    let encoded = RefCell::new(Vec::<String>::new());
    let request = policy
        .prepare_text(&body, 1024, |text| {
            assert!(text.len() < 1024);
            assert!(encoded.borrow().len() < 16);
            encoded.borrow_mut().push(text.to_owned());
            // Byte tokenizer is only a finite preparation oracle; not a real
            // tokenizer/model parity claim. Single-letter native labels are exact.
            Ok(text.bytes().map(u32::from).collect())
        })
        .unwrap();
    let encoded = encoded.into_inner();
    let prefix = "<|startoftext|><|im_start|>user\nshared\n\n\nQUESTION:\n";
    assert_eq!(
        encoded.first().unwrap(),
        prefix,
        "complete state prefix changed"
    );
    let prompts: Vec<_> = encoded
        .iter()
        .filter(|text| text.starts_with(prefix) && text.len() > prefix.len())
        .collect();
    assert_eq!(
        prompts.len(),
        2,
        "each question must encode one whole prompt"
    );
    assert!(
        prompts[0].contains("first\n\nOptions:\nB bee\nA aye"),
        "first question or criterion order changed"
    );
    assert!(
        prompts[1].contains("second\n\nOptions:\nA aye\nB bee"),
        "second question or criterion order changed"
    );
    assert_eq!(
        request.input_tokens(),
        prompts.iter().map(|text| text.len() as u64).sum::<u64>()
    );
    assert_eq!(
        request.max_prompt_tokens(),
        prompts.iter().map(|text| text.len()).max().unwrap()
    );
    assert!(request
        .trunk_tokens()
        .starts_with(&prefix.bytes().map(u32::from).collect::<Vec<_>>()));
    let question: crate::decision::Question<()> = crate::decision::Question::Noul {
        instructions: (),
        criteria: None,
    };
    assert_eq!(f64::from(policy.calibration.temperature(&question)), 2.5); // Caller policy, not execution calibration proof.
    Ok(())
}

#[test]
fn explicit_refusals_precede_token_preparation_without_defaults() -> Result<()> {
    let cases = [
        (
            2,
            StateNullability::RequireValue,
            r#"{"state":null,"questions":{"q":{"type":"choice","instructions":"x","criteria":{"A":null,"B":null}}}}"#,
            "\"state\" must be provided",
            false,
        ),
        (
            0,
            StateNullability::AllowNull,
            r#"{"state":null,"questions":{"q":{"type":"choice","instructions":"x","criteria":{"A":null,"B":null}}}}"#,
            "at most 0",
            false,
        ),
        (
            2,
            StateNullability::AllowNull,
            r#"{"state":"shared","images":["data:image/png;base64,A"],"questions":{"q":{"type":"choice","instructions":"x","criteria":{"A":null,"B":null}}}}"#,
            "media encoder",
            true,
        ),
    ];
    for (limit, nullability, text, expected, unsupported) in cases {
        let policy = policy(SOURCE, limit, nullability)?;
        let calls_ref = std::cell::Cell::new(0usize);
        let error = match policy.prepare_text(&body(text), 1024, |_| {
            calls_ref.set(calls_ref.get() + 1);
            Ok(vec![0])
        }) {
            Err(error) => error,
            Ok(_) => panic!("explicit refusal became success"),
        };
        assert_eq!(calls_ref.get(), 0, "refusal reached token preparation");
        assert_eq!(
            matches!(&error, SystemOneError::Unsupported(_)),
            unsupported
        );
        assert!(error.to_string().contains(expected), "{error}");
    }
    Ok(())
}

#[test]
fn uncertified_template_keeps_ordinary_rendering_without_invented_certificate() -> Result<()> {
    let policy = policy("ordinary", 2, StateNullability::AllowNull)?;
    assert_eq!(policy.template.kind(), SystemOneTemplateKind::D1Causal);
    assert!(!policy.template.has_causal_certificate());
    let question = crate::decision::Question::Choice {
        instructions: OrderedJson::String("choose".into()),
        criteria: vec![("A".into(), None), ("B".into(), None)],
    };
    assert_eq!(
        policy.template.render(
            "q",
            &OrderedJson::Null,
            &question,
            &["A".into(), "B".into()],
            crate::decision::SystemOneMedia::Text
        )?,
        "ordinary"
    );
    let request = body(
        r#"{"state":null,"questions":{"q":{"type":"choice","instructions":"choose","criteria":{"A":null,"B":null}}}}"#,
    );
    let calls = std::cell::Cell::new(0usize);
    let error = match policy.prepare_text(&request, 1024, |_| {
        calls.set(calls.get() + 1);
        Ok(vec![0])
    }) {
        Err(error) => error,
        Ok(_) => panic!("unknown source received a shared-trunk certificate"),
    };
    assert_eq!(calls.get(), 0);
    assert!(matches!(&error, SystemOneError::InvalidRequest(_)));
    assert!(
        error
            .to_string()
            .contains("no audited state-prefix certificate"),
        "{error}"
    );
    Ok(())
}

#[test]
fn native_policy_binds_a_d1_checkpoint_and_nothing_else() -> Result<()> {
    let dir = std::env::temp_dir().join(format!("lloom-causal-native-{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    let limits = (4, NoulAliases::TrueFalseOnly, StateNullability::AllowNull);
    std::fs::write(
        dir.join("config.json"),
        r#"{"model_type":"lfm2_vl","auto_map":{"AutoModel":"modeling_d1.D1Model"}}"#,
    )?;
    let policy = CausalPolicy::native(&dir, limits.0, limits.1, limits.2)?
        .expect("d1-3B safetensors binds its engine-shipped template");
    assert!(policy.template.has_causal_certificate());
    assert_eq!(policy.max_options, 4);
    std::fs::write(dir.join("config.json"), r#"{"model_type":"lfm2_vl"}"#)?;
    assert!(CausalPolicy::native(&dir, limits.0, limits.1, limits.2)?.is_none());
    std::fs::remove_dir_all(&dir)?;
    Ok(())
}
