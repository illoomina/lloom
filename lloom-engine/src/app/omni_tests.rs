//! Actual shared preparation/readout and marker-only recovery-entry controls.
//! V16, allowance64, <=3 questions/control, no Tensor/native/Loaded constructor.
use super::*;
use std::collections::BTreeMap;
use tokenizers::AddedToken;
fn text() -> Text {
    let words = [
        "[UNK]",
        "<|startoftext|>",
        "<|reserved_7|>",
        "<|reserved_8|>",
        "<|reserved_9|>",
        "<|mask|>",
        "<|reserved_10|>",
        "<|reserved_11|>",
        "A",
        "B",
        "yes",
        "no",
        "choose",
        "only",
        "low",
        "high",
    ];
    let model = tokenizers::models::wordlevel::WordLevel::builder()
        .vocab(
            words
                .iter()
                .enumerate()
                .map(|(i, s)| (s.to_string(), i as u32))
                .collect(),
        )
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(
        tokenizers::pre_tokenizers::whitespace::WhitespaceSplit,
    ));
    tokenizer.add_special_tokens(
        &words[1..8]
            .iter()
            .map(|s| AddedToken::from(*s, true))
            .collect::<Vec<_>>(),
    );
    assert_eq!(tokenizer.get_vocab_size(true), 16);
    assert_eq!(tokenizer.token_to_id("<|mask|>"), Some(5));
    Text {
        tokenizer,
        marker: 5,
    }
}
fn policy() -> Policy {
    Policy {
        template: SystemOneTemplate::compile(
            include_str!("../decision/fixtures/d1-omni-systemone.jinja"),
            SystemOneTemplateKind::D1Omni,
        )
        .unwrap(),
        max_options: 3,
        noul_aliases: NoulAliases::TrueFalseAndYesNo,
        state_nullability: StateNullability::AllowNull,
        media_marker: "<media>".into(),
    }
}
fn facts(temperatures: &BTreeMap<String, f64>) -> OmniMetadata<'_> {
    OmniMetadata {
        vocab_size: 16,
        max_length: 128,
        image_text_length: 64,
        audio_text_length: 64,
        temperatures,
    }
}
fn body(s: &str) -> OrderedJson {
    OrderedJson::from_slice(s.as_bytes()).unwrap()
}
#[test]
fn actual_raw_json_preparation_and_endpoint_singleton_preserve_reference_split() -> Result<()> {
    let text = text();
    let policy = policy();
    let temperatures = BTreeMap::from([("choice:2".into(), 2.)]);
    let state = body(r#"{"z":false,"a":[null,17]}"#);
    let request = body(
        r#"{"questions":{"z":{"type":"choice","instructions":{"z":false,"a":17},"criteria":{"B":null,"A":{"z":1,"a":2}}},"s":{"type":"score","instructions":"choose","criteria":[{"z":false,"a":[null]},17]},"one":{"type":"choice","instructions":false,"criteria":{"only":null}}}}"#,
    );
    let questions = parse_systemone_questions(&request, 3, NoulAliases::TrueFalseAndYesNo)?;
    let (rows, readouts) = prepare_questions(
        &text,
        &policy,
        &state,
        &questions,
        OmniMedia::Text,
        SystemOneMedia::Text,
        "",
        64,
        facts(&temperatures),
    )
    .unwrap();
    assert_eq!(rows.len(), 3);
    assert_eq!(
        rows.iter().map(|r| r.question_type).collect::<Vec<_>>(),
        [0, 1, 0]
    );
    assert_eq!(
        rows.iter().map(|r| r.markers.len()).collect::<Vec<_>>(),
        [2, 2, 1]
    );
    for row in &rows {
        assert!(row.ids.len() <= 32);
        for &m in &row.markers {
            assert_eq!(row.ids[m], 5);
        }
    }
    let answer = questions[1]
        .1
        .systemone_answer(&readouts[1].probabilities(&[0., 0.])?)?;
    assert!(answer
        .to_json_string()
        .unwrap()
        .contains(r#""legend":{"0":{"z":false,"a":[null]},"1":17}"#));
    assert_eq!(readouts[2].probabilities(&[9., f32::NAN])?, [1.]);
    assert_eq!(
        questions[2].1.systemone_answer(&[1.])?.to_json()["confidence"],
        1.
    );
    let legacy = Question::Choice {
        instructions: "choose".into(),
        criteria: vec![("only".into(), None)],
    };
    assert!(
        OmniReadout::prepare(&legacy, OmniMedia::Text, &BTreeMap::new()).is_err(),
        "endpoint must not weaken Python singleton refusal"
    );
    let rendered =
        policy
            .template
            .render("z", &state, &questions[0].1, &[], SystemOneMedia::Text)?;
    assert!(rendered.contains(r#"{"z": false, "a": [null, 17]}"#));
    assert!(rendered.contains(r#"{"z": false, "a": 17}"#));
    assert!(rendered.find(" B").unwrap() < rendered.find(" A").unwrap());
    let ordinary = SystemOneTemplate::compile(
        "{{ state | tojson }}|{{ instructions | tojson }}",
        SystemOneTemplateKind::D1Omni,
    )?;
    assert_eq!(
        ordinary.render("q", &state, &questions[2].1, &[], SystemOneMedia::Text)?,
        r#"{"z": false, "a": [null, 17]}|false"#
    );
    // This unknown source remains ordinary-renderable; no marked-layout proof
    // or successful Loaded/Engine pass is inferred from the host control.
    Ok(())
}
#[test]
fn actual_media_preparation_and_noul_readout_keep_text_relative_markers() -> Result<()> {
    let text = text();
    let policy = policy();
    let temperatures = BTreeMap::from([("noul:2".into(), 2.)]);
    let request = body(
        r#"{"questions":{"n":{"type":"noul","instructions":"choose","criteria":{"true":{"z":false},"false":null}}}}"#,
    );
    let questions = parse_systemone_questions(&request, 3, NoulAliases::TrueFalseOnly)?;
    let (plain, text_readout) = prepare_questions(
        &text,
        &policy,
        &OrderedJson::Null,
        &questions,
        OmniMedia::Text,
        SystemOneMedia::Text,
        "",
        64,
        facts(&temperatures),
    )
    .unwrap();
    let p = text_readout[0].probabilities(&[0., 2.])?;
    assert!((p[0] - 0.7310585786300049).abs() < 1e-14);
    for (kind, rendering, expected) in [
        (
            OmniMedia::Image,
            SystemOneMedia::Images {
                count: 2,
                marker: "<media>",
            },
            "<media><media>",
        ),
        (
            OmniMedia::Audio,
            SystemOneMedia::Audio { marker: "<media>" },
            "<media>",
        ),
    ] {
        let (rows, readouts) = prepare_questions(
            &text,
            &policy,
            &OrderedJson::Null,
            &questions,
            kind,
            rendering,
            expected,
            64,
            facts(&temperatures),
        )
        .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].question_type, 2);
        assert_eq!(rows[0].markers.len(), 2);
        assert!(rows[0].ids.len() <= 24);
        for &m in &rows[0].markers {
            assert_eq!(rows[0].ids[m], 5);
        }
        assert_eq!(
            rows[0].ids[0], 1,
            "media markers must not be tokenized into text rows"
        );
        let p = readouts[0].probabilities(&[0., 2., f32::NAN])?;
        assert!((p[0] - 0.8807970779778823).abs() < 1e-14);
        assert_eq!(questions[0].1.systemone_answer(&p)?.to_json()["noul"], p[0]);
        assert!(prepare_questions(
            &text,
            &policy,
            &OrderedJson::Null,
            &questions,
            kind,
            rendering,
            "wrong",
            64,
            facts(&temperatures)
        )
        .is_err());
    }
    assert_eq!(plain[0].ids[0], 1);
    let bad = OmniMetadata {
        vocab_size: 5,
        ..facts(&temperatures)
    };
    assert!(prepare_questions(
        &text,
        &policy,
        &OrderedJson::Null,
        &questions,
        OmniMedia::Text,
        SystemOneMedia::Text,
        "",
        64,
        bad
    )
    .is_err());
    Ok(())
}
struct Probe {
    calls: usize,
    mode: u8,
}
impl Recovery for Probe {
    fn before_reuse(&mut self, _: OmniMetadata<'_>) -> Result<()> {
        self.calls += 1;
        match self.mode {
            1 => lloom_tensor::bail!("intentional recovery refusal"),
            2 => panic!("intentional recovery panic"),
            _ => Ok(()),
        }
    }
}
#[test]
fn actual_recovery_entry_refusal_unwind_retry_and_success_keep_marker_custody() -> Result<()> {
    let temperatures = BTreeMap::new();
    let mut recovery = Probe { calls: 0, mode: 0 };
    let mut incomplete = false;
    {
        let mut work = begin_work(facts(&temperatures), &mut recovery, &mut incomplete)?;
        work.complete = true;
    }
    assert!(!incomplete);
    assert_eq!(recovery.calls, 0, "successful fresh work must not fence");
    {
        let _work = begin_work(facts(&temperatures), &mut recovery, &mut incomplete)?;
    }
    assert!(incomplete);
    assert_eq!(recovery.calls, 0);
    recovery.mode = 1;
    let refusal = begin_work(facts(&temperatures), &mut recovery, &mut incomplete)
        .err()
        .unwrap();
    assert!(refusal.to_string().contains("intentional recovery refusal"));
    assert!(incomplete);
    assert_eq!(recovery.calls, 1);
    recovery.mode = 2;
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _work = begin_work(facts(&temperatures), &mut recovery, &mut incomplete).unwrap();
        panic!("recovery unexpectedly returned")
    }))
    .unwrap_err();
    assert_eq!(
        panic.downcast_ref::<&str>(),
        Some(&"intentional recovery panic")
    );
    assert!(incomplete);
    assert_eq!(recovery.calls, 2);
    recovery.mode = 0;
    {
        let mut work = begin_work(facts(&temperatures), &mut recovery, &mut incomplete)?;
        work.complete = true;
    }
    assert!(!incomplete);
    assert_eq!(
        recovery.calls, 3,
        "retry must recover once before started work"
    );
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _work = begin_work(facts(&temperatures), &mut recovery, &mut incomplete).unwrap();
        panic!("intentional attention panic")
    }))
    .unwrap_err();
    assert_eq!(
        panic.downcast_ref::<&str>(),
        Some(&"intentional attention panic")
    );
    assert!(incomplete);
    assert_eq!(recovery.calls, 3);
    Ok(())
}

#[test]
fn native_policy_takes_the_omni_template_from_the_checkpoint() {
    let dir = std::env::temp_dir().join(format!("lloom-omni-native-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("config.json"), r#"{"model_type":"d1_omni"}"#).unwrap();
    let policy = Policy::native(
        &dir,
        8,
        NoulAliases::TrueFalseOnly,
        StateNullability::AllowNull,
        "<image>".into(),
    )
    .unwrap();
    assert_eq!(policy.template.kind(), SystemOneTemplateKind::D1Omni);
    std::fs::write(dir.join("config.json"), r#"{"model_type":"lfm2"}"#).unwrap();
    let refused = Policy::native(
        &dir,
        8,
        NoulAliases::TrueFalseOnly,
        StateNullability::AllowNull,
        "<image>".into(),
    );
    assert!(refused.is_err(), "a plain checkpoint is not d1-omni");
    std::fs::remove_dir_all(&dir).unwrap();
}
