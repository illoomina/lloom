//! Actual renderer/prefix + incumbent expansion only; no decoder, tensor,
//! model, Loaded/native or admission proof. Two questions/two placeholders,
//! three whole prompt encodes per case, each <1024 bytes/tokens.
use super::*;
use crate::app::vision::expand_image_tokens;
use crate::decision::{SystemOneMedia, SystemOneTemplate, SystemOneTemplateKind};
use std::cell::{Cell, RefCell};

const SOURCE: &str = include_str!("../fixtures/d1-systemone.jinja");
const HEAD: &str = "<|startoftext|><|im_start|>user\n";
const TAIL: &str =
    "\n\nOptions:\nA A\nB B\n\nReply with the option code only.<|im_end|>\n<|im_start|>assistant\n";
fn template() -> Result<SystemOneTemplate> {
    SystemOneTemplate::compile(SOURCE, SystemOneTemplateKind::D1Causal)
}
fn question(instructions: &str) -> Question<OrderedJson> {
    Question::Choice {
        instructions: OrderedJson::String(instructions.into()),
        criteria: vec![("A".into(), None), ("B".into(), None)],
    }
}
fn questions() -> Result<Vec<(String, Question<OrderedJson>, Temperature)>> {
    Ok(vec![
        ("z".into(), question("first"), 1.0.try_into()?),
        ("a".into(), question("second"), 1.0.try_into()?),
    ])
}
fn label(text: &str) -> Result<Vec<u32>> {
    match text.trim() {
        "A" => Ok(vec![65]),
        "B" => Ok(vec![66]),
        _ => lloom_tensor::bail!("unexpected literal label {text:?}"),
    }
}
fn captured(
    state: &OrderedJson,
    media: SystemOneMedia<'_>,
    expansions: &[String],
    legacy: bool,
) -> Result<(CausalRequest, Vec<String>)> {
    assert!(expansions.len() <= 2);
    let prompts = RefCell::new(Vec::new());
    let label_calls = Cell::new(0);
    let encode_label = |text: &str| {
        label_calls.set(label_calls.get() + 1);
        assert!(label_calls.get() <= 8);
        // Labels never enter placeholder expansion or the prompt encoder.
        assert!(!text.contains("<image>"));
        label(text)
    };
    let encode_prompt = |text: &str| {
        assert!(text.len() < 1024);
        assert!(prompts.borrow().len() < 3);
        let whole = expand_image_tokens(text, expansions).map_err(Error::wrap)?;
        assert!(whole.len() < 1024);
        let tokens = whole.bytes().map(u32::from).collect();
        prompts.borrow_mut().push(whole);
        Ok(tokens)
    };
    let template = template()?;
    let request = if legacy {
        CausalRequest::render(
            &template,
            state,
            questions()?,
            1024,
            encode_label,
            encode_prompt,
        )?
    } else {
        CausalRequest::render_with_media(
            &template,
            state,
            questions()?,
            1024,
            media,
            encode_label,
            encode_prompt,
        )?
    };
    assert_eq!(label_calls.get(), 8);
    Ok((request, prompts.into_inner()))
}
fn assert_prompts(request: &CausalRequest, prompts: &[String], prefix: &str) {
    assert_eq!(
        prompts,
        &[
            prefix.to_owned(),
            format!("{prefix}first{TAIL}"),
            format!("{prefix}second{TAIL}")
        ]
    );
    assert_eq!(request.tasks.len(), 2);
    for (task, text) in request.tasks.iter().zip(&prompts[1..]) {
        assert_eq!(task.prompt, text.bytes().map(u32::from).collect::<Vec<_>>());
    }
    assert!(request
        .trunk_tokens()
        .starts_with(&prefix.bytes().map(u32::from).collect::<Vec<_>>()));
    assert_eq!(
        request.input_tokens(),
        prompts[1..].iter().map(|p| p.len() as u64).sum::<u64>()
    );
}

#[test]
fn legacy_text_bytes_and_two_distinct_whole_image_expansions_match() -> Result<()> {
    let state = OrderedJson::String("shared".into());
    let (legacy, old) = captured(&state, SystemOneMedia::Text, &[], true)?;
    let (text, new) = captured(&state, SystemOneMedia::Text, &[], false)?;
    let prefix = format!("{HEAD}shared\n\n\nQUESTION:\n");
    assert_prompts(&legacy, &old, &prefix);
    assert_prompts(&text, &new, &prefix);
    assert_eq!(new, old, "legacy Text bytes changed");
    assert_eq!(text.trunk_tokens(), legacy.trunk_tokens());
    let expansions = ["<A>".to_owned(), "<BB>".to_owned()];
    let (images, encoded) = captured(
        &state,
        SystemOneMedia::Images {
            count: 2,
            marker: "<image>",
        },
        &expansions,
        false,
    )?;
    assert_prompts(
        &images,
        &encoded,
        &format!("{HEAD}<A><BB>shared\n\n\nQUESTION:\n"),
    );
    Ok(())
}

#[test]
fn marker_state_values_null_and_marker_keys_keep_or_refuse_exact_certificate() -> Result<()> {
    let media = SystemOneMedia::Images {
        count: 2,
        marker: "<image>",
    };
    let expansions = ["<A>".to_owned(), "<BB>".to_owned()];
    let state = OrderedJson::String("before<image>after".into());
    let (request, encoded) = captured(&state, media, &expansions, false)?;
    assert_prompts(
        &request,
        &encoded,
        &format!("{HEAD}<A><BB>before after\n\n\nQUESTION:\n"),
    );
    let structured = OrderedJson::Object(vec![(
        "key".into(),
        OrderedJson::String("before<image>after".into()),
    )]);
    let (request, encoded) = captured(&structured, media, &expansions, false)?;
    assert_prompts(
        &request,
        &encoded,
        &format!("{HEAD}<A><BB>{{\n  \"key\": \"before after\"\n}}\n\n\nQUESTION:\n"),
    );
    let (request, encoded) = captured(&OrderedJson::Null, media, &expansions, false)?;
    assert_prompts(&request, &encoded, &format!("{HEAD}<A><BB>"));
    // transform_json intentionally leaves JSON keys untouched. An extra marker
    // in a key must make the actual expander refuse, not consume another image.
    let keyed = OrderedJson::Object(vec![(
        "<image>".into(),
        OrderedJson::String("<image>".into()),
    )]);
    let template = template()?;
    let label_calls = Cell::new(0);
    let prompt_calls = Cell::new(0);
    let error = CausalRequest::render_with_media(
        &template,
        &keyed,
        questions()?,
        1024,
        media,
        |s| {
            label_calls.set(label_calls.get() + 1);
            label(s)
        },
        |s| {
            prompt_calls.set(prompt_calls.get() + 1);
            expand_image_tokens(s, &expansions)
                .map(|p| p.bytes().map(u32::from).collect())
                .map_err(Error::wrap)
        },
    )
    .err()
    .expect("marker-bearing key silently rebound an image");
    assert!(
        error
            .to_string()
            .contains("3 `<image>` token(s) for 2 attached image(s)"),
        "{error}"
    );
    assert_eq!(prompt_calls.get(), 1);
    assert_eq!(label_calls.get(), 0);
    Ok(())
}

#[test]
fn invalid_media_unknown_certificate_and_whole_token_boundary_refuse() -> Result<()> {
    let template = template()?;
    let state = OrderedJson::String("shared".into());
    for media in [
        SystemOneMedia::Images {
            count: 0,
            marker: "<image>",
        },
        SystemOneMedia::Images {
            count: 9,
            marker: "<image>",
        },
        SystemOneMedia::Images {
            count: 1,
            marker: "",
        },
    ] {
        let calls = Cell::new(0);
        let error = CausalRequest::render_with_media(
            &template,
            &state,
            questions()?,
            1024,
            media,
            |_| {
                calls.set(calls.get() + 1);
                Ok(vec![65])
            },
            |_| {
                calls.set(calls.get() + 1);
                Ok(vec![1])
            },
        )
        .err()
        .expect("invalid media became success");
        assert!(error
            .to_string()
            .contains("images require 1..=8 items and a nonempty marker"));
        assert_eq!(calls.get(), 0, "invalid media reached token encoders");
    }
    let unknown = SystemOneTemplate::compile(
        "ordinary {{ images | join('-') }}",
        SystemOneTemplateKind::D1Causal,
    )?;
    assert!(!unknown.has_causal_certificate());
    let media = SystemOneMedia::Images {
        count: 2,
        marker: "<image>",
    };
    assert_eq!(
        unknown.render(
            "q",
            &state,
            &question("first"),
            &["A".into(), "B".into()],
            media
        )?,
        "ordinary <image>-<image>"
    );
    let calls = Cell::new(0);
    let error = CausalRequest::render_with_media(
        &unknown,
        &state,
        questions()?,
        1024,
        media,
        |_| {
            calls.set(calls.get() + 1);
            Ok(vec![65])
        },
        |_| {
            calls.set(calls.get() + 1);
            Ok(vec![1])
        },
    )
    .err()
    .expect("unknown source received a media certificate");
    assert!(error
        .to_string()
        .contains("no audited state-prefix certificate"));
    assert_eq!(calls.get(), 0);
    let expansions = ["<A>".to_owned(), "<BB>".to_owned()];
    let calls = Cell::new(0);
    let error = CausalRequest::render_with_media(
        &template,
        &state,
        vec![("q".into(), question("first"), 1.0.try_into()?)],
        1024,
        media,
        label,
        |s| {
            let expanded = expand_image_tokens(s, &expansions).map_err(Error::wrap)?;
            assert!(expanded.len() < 1024);
            let mut ids: Vec<_> = expanded.bytes().map(u32::from).collect();
            let call = calls.get();
            calls.set(call + 1);
            assert!(call < 2);
            if call == 0 {
                *ids.last_mut().unwrap() = 999;
            }
            Ok(ids)
        },
    )
    .err()
    .expect("incompatible whole-token boundary became success");
    assert!(
        error
            .to_string()
            .contains("causal state prefix must be shared completely"),
        "{error}"
    );
    assert_eq!(calls.get(), 2);
    Ok(())
}
