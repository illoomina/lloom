//! Ordered systemone state/media extraction, before decoding or model rendering.
//! Media content determines image versus audio; a data URL's field is not proof.
use super::OrderedJson;
use lloom_tensor::{bail, Result};
use std::borrow::Cow;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StateNullability {
    AllowNull,
    RequireValue,
}

/// Borrowed encoded input; decoding and its byte/shape admission are separate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MediaSource<'a> {
    DataUrl(&'a str),
    AudioBase64(&'a str),
}

#[derive(Clone, Debug, PartialEq)]
pub struct ParsedState<'a> {
    pub state: Cow<'a, OrderedJson>,
    pub media: Vec<MediaSource<'a>>,
}

/// Extract media in reference order: files, images, then message parts. State
/// and descriptions remain ordered JSON. Media-bearing state is copied without
/// the removed parts: their encoded payloads stay borrowed, never transiently
/// cloned. Other retained JSON keys/values are copied. The caller still admits
/// original JSON, transformed state, decoded bytes and subsequent tensor work.
/// Directly constructed objects may have duplicate keys: only the first key
/// selected by OrderedJson::get is transformed; later entries remain unchanged.
/// This function performs no file/network access or base64/image/audio decode.
pub fn parse_systemone_state(
    body: &OrderedJson,
    nullability: StateNullability,
) -> Result<ParsedState<'_>> {
    let Some(state) = body.get("state") else {
        bail!("\"state\" must be provided")
    };
    if nullability == StateNullability::RequireValue && matches!(state, OrderedJson::Null) {
        bail!("\"state\" must be provided")
    }
    if body.get("videos").is_some_and(|value| !empty(value)) {
        bail!("\"videos\" is not supported")
    }
    let mut media = Vec::new();
    for key in ["files", "images"] {
        match body.get(key) {
            None | Some(OrderedJson::Null) => (),
            Some(OrderedJson::Array(urls)) => {
                for url in urls {
                    push_media(&mut media, url, false)?;
                }
            }
            _ => bail!("\"{key}\" must be an array"),
        }
    }
    let wrapped = state.get("messages").is_some();
    let messages = if wrapped {
        state.get("messages").expect("checked object key")
    } else {
        state
    };
    let OrderedJson::Array(messages) = messages else {
        return Ok(ParsedState {
            state: Cow::Borrowed(state),
            media,
        });
    };
    // At most eight removals, the same bound as retained media references.
    let mut removed = Vec::new();
    for (message_index, message) in messages.iter().enumerate() {
        let Some(OrderedJson::Array(parts)) = message.get("content") else {
            continue;
        };
        for (part_index, part) in parts.iter().enumerate() {
            let value = match part.get("type").and_then(OrderedJson::as_str) {
                Some("image_url") => part
                    .get("image_url")
                    .map(|value| (value.get("url").unwrap_or(value), false)),
                Some("input_audio") => part.get("input_audio").map(|value| {
                    (
                        value
                            .get("data")
                            .or_else(|| value.get("url"))
                            .unwrap_or(&OrderedJson::Null),
                        true,
                    )
                }),
                _ => None,
            };
            if let Some((value, audio)) = value {
                push_media(&mut media, value, audio)?;
                removed.push((message_index, part_index));
            }
        }
    }
    if removed.is_empty() {
        return Ok(ParsedState {
            state: Cow::Borrowed(state),
            media,
        });
    }
    // Construct the filtered tree directly. Cloning state then removing parts
    // would transiently duplicate every encoded media payload.
    let filtered = OrderedJson::Array(
        messages
            .iter()
            .enumerate()
            .map(|(message_index, message)| {
                if !removed.iter().any(|&(m, _)| m == message_index) {
                    return message.clone();
                }
                let OrderedJson::Object(entries) = message else {
                    unreachable!("media part requires object message")
                };
                let content_index = entries
                    .iter()
                    .position(|(key, _)| key == "content")
                    .expect("selected content key");
                OrderedJson::Object(
                    entries
                        .iter()
                        .enumerate()
                        .map(|(entry_index, (key, value))| {
                            let copied = if entry_index == content_index {
                                let OrderedJson::Array(parts) = value else {
                                    unreachable!("media part requires content array")
                                };
                                OrderedJson::Array(
                                    parts
                                        .iter()
                                        .enumerate()
                                        .filter(|(p, _)| !removed.contains(&(message_index, *p)))
                                        .map(|(_, part)| part.clone())
                                        .collect(),
                                )
                            } else {
                                value.clone()
                            };
                            (key.clone(), copied)
                        })
                        .collect(),
                )
            })
            .collect(),
    );
    let transformed = if wrapped {
        let OrderedJson::Object(entries) = state else {
            unreachable!("messages key requires object state")
        };
        // Move only the first selected key; public Vec-backed objects can
        // contain duplicates even though the serde parser canonicalizes them.
        let messages_index = entries
            .iter()
            .position(|(key, _)| key == "messages")
            .expect("selected messages key");
        let mut filtered = Some(filtered);
        OrderedJson::Object(
            entries
                .iter()
                .enumerate()
                .map(|(entry_index, (key, value))| {
                    let copied = if entry_index == messages_index {
                        filtered.take().expect("selected messages key visited once")
                    } else {
                        value.clone()
                    };
                    (key.clone(), copied)
                })
                .collect(),
        )
    } else {
        filtered
    };
    Ok(ParsedState {
        state: Cow::Owned(transformed),
        media,
    })
}

fn empty(value: &OrderedJson) -> bool {
    match value {
        OrderedJson::Null => true,
        OrderedJson::Array(v) => v.is_empty(),
        OrderedJson::Object(v) => v.is_empty(),
        // JSON primitives have one element, including an empty string.
        OrderedJson::String(_) | OrderedJson::Bool(_) | OrderedJson::Number(_) => false,
    }
}
fn push_media<'a>(
    out: &mut Vec<MediaSource<'a>>,
    value: &'a OrderedJson,
    audio: bool,
) -> Result<()> {
    let Some(text) = value.as_str() else {
        if audio {
            bail!("audio must be base64 data")
        }
        bail!("images must be data URLs (data:image/...;base64,... or data:audio/...;base64,...)")
    };
    if audio {
        if text.starts_with("http") || text.starts_with("file://") {
            bail!("audio must be base64 data")
        }
    } else if !text.starts_with("data:") {
        bail!("images must be data URLs (data:image/...;base64,... or data:audio/...;base64,...)")
    }
    if out.len() >= 8 {
        bail!("too many media files, the maximum is 8")
    }
    out.push(if audio {
        MediaSource::AudioBase64(text)
    } else {
        MediaSource::DataUrl(text)
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn json(text: &str) -> OrderedJson {
        OrderedJson::from_slice(text.as_bytes()).unwrap()
    }

    #[test]
    fn media_order_and_ordered_state_transformation_match_reference() -> Result<()> {
        let body = json(
            r#"{"images":["data:image/second;base64,B"],"files":["data:audio/first;base64,A"],"state":{"z":0,"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/third;base64,C"}},{"type":"text","text":"雪\nline"},{"type":"input_audio","input_audio":{"data":"RAW_D","url":"ignored"}},{"type":"input_audio","input_audio":{"url":"RAW_E"}},{"type":"image_url"}]},7],"a":{"z":2,"b":1}}}"#,
        );
        let before = serde_json::to_string(&body).unwrap();
        let parsed = parse_systemone_state(&body, StateNullability::AllowNull)?;
        assert_eq!(
            parsed.media,
            [
                MediaSource::DataUrl("data:audio/first;base64,A"),
                MediaSource::DataUrl("data:image/second;base64,B"),
                MediaSource::DataUrl("data:image/third;base64,C"),
                MediaSource::AudioBase64("RAW_D"),
                MediaSource::AudioBase64("RAW_E"),
            ]
        );
        assert!(matches!(&parsed.state, Cow::Owned(_)));
        assert_eq!(
            serde_json::to_string(parsed.state.as_ref()).unwrap(),
            r#"{"z":0,"messages":[{"role":"user","content":[{"type":"text","text":"雪\nline"},{"type":"image_url"}]},7],"a":{"z":2,"b":1}}"#
        );
        assert_eq!(serde_json::to_string(&body).unwrap(), before);
        let OrderedJson::Array(files) = body.get("files").unwrap() else {
            panic!("files")
        };
        let MediaSource::DataUrl(first) = parsed.media[0] else {
            panic!("url")
        };
        assert_eq!(first.as_ptr(), files[0].as_str().unwrap().as_ptr());
        let plain = json(
            r#"{"state":[{"content":[{"type":"image_url","image_url":"data:image/x;base64,X"},{"type":"text","text":"kept"}]}]}"#,
        );
        let parsed = parse_systemone_state(&plain, StateNullability::RequireValue)?;
        assert_eq!(
            serde_json::to_string(parsed.state.as_ref()).unwrap(),
            r#"[{"content":[{"type":"text","text":"kept"}]}]"#
        );
        // Direct embedding callers can construct duplicate object entries.
        // Transform the first selected content/messages keys only; retain later
        // nonarray duplicates rather than panicking or silently dropping them.
        let media_part = json(r#"{"type":"image_url","image_url":"data:image/x;base64,X"}"#);
        let direct = OrderedJson::Object(vec![(
            "state".into(),
            OrderedJson::Object(vec![
                (
                    "messages".into(),
                    OrderedJson::Array(vec![OrderedJson::Object(vec![
                        ("content".into(), OrderedJson::Array(vec![media_part])),
                        ("content".into(), OrderedJson::Bool(false)),
                    ])]),
                ),
                ("messages".into(), OrderedJson::Null),
            ]),
        )]);
        let parsed = parse_systemone_state(&direct, StateNullability::AllowNull)?;
        assert_eq!(parsed.media.len(), 1);
        assert_eq!(
            serde_json::to_string(parsed.state.as_ref()).unwrap(),
            r#"{"messages":[{"content":[],"content":false}],"messages":null}"#
        );
        let first_scalar = OrderedJson::Object(vec![(
            "state".into(),
            OrderedJson::Object(vec![
                ("messages".into(), OrderedJson::Null),
                ("messages".into(), OrderedJson::Array(vec![])),
            ]),
        )]);
        let parsed = parse_systemone_state(&first_scalar, StateNullability::AllowNull)?;
        assert!(matches!(parsed.state, Cow::Borrowed(_)));
        assert!(parsed.media.is_empty());
        for kind in ["null", "7", "false", "[]", "{}"] {
            let body = json(&format!(
                r#"{{"state":[{{"content":[{{"type":{kind},"image_url":"data:image/x;base64,X"}}]}}]}}"#
            ));
            let parsed = parse_systemone_state(&body, StateNullability::AllowNull)?;
            assert!(matches!(parsed.state, Cow::Borrowed(_)));
            assert!(parsed.media.is_empty());
        }
        Ok(())
    }

    #[test]
    fn state_nullability_and_json_empty_are_explicit() -> Result<()> {
        assert!(parse_systemone_state(&json("{}"), StateNullability::AllowNull).is_err());
        for state in [
            "null",
            "0",
            "false",
            r#""""#,
            "[]",
            "{}",
            r#"{"messages":null}"#,
        ] {
            let body = json(&format!(r#"{{"state":{state}}}"#));
            let parsed = parse_systemone_state(&body, StateNullability::AllowNull)?;
            assert!(matches!(parsed.state, Cow::Borrowed(_)));
            assert_eq!(parsed.state.as_ref(), body.get("state").unwrap());
            assert!(parsed.media.is_empty());
            assert_eq!(
                parse_systemone_state(&body, StateNullability::RequireValue).is_ok(),
                state != "null"
            );
        }
        for videos in ["null", "[]", "{}"] {
            let body = json(&format!(r#"{{"state":null,"videos":{videos}}}"#));
            parse_systemone_state(&body, StateNullability::AllowNull)?;
        }
        // nlohmann JSON.empty(), not the primitive string/container's empty().
        for videos in [r#""""#, "false", "0", "[null]", r#"{"x":0}"#] {
            let body = json(&format!(r#"{{"state":null,"videos":{videos}}}"#));
            assert!(parse_systemone_state(&body, StateNullability::AllowNull)
                .unwrap_err()
                .to_string()
                .contains("videos"));
        }
        Ok(())
    }

    #[test]
    fn media_limits_and_invalid_encoded_sources_refuse_before_decoding() -> Result<()> {
        for count in [8, 9] {
            let urls = std::iter::repeat_n(r#""data:image/x;base64,X""#, count)
                .collect::<Vec<_>>()
                .join(",");
            let body = json(&format!(r#"{{"state":null,"files":[{urls}]}}"#));
            let parsed = parse_systemone_state(&body, StateNullability::AllowNull);
            if count == 8 {
                assert_eq!(parsed?.media.len(), 8);
            } else {
                assert!(parsed.unwrap_err().to_string().contains("maximum is 8"));
            }
        }
        for suffix in [
            r#""files":{}"#,
            r#""images":[null]"#,
            r#""files":["https://example.invalid/x"]"#,
            r#""images":["file:///private/x"]"#,
        ] {
            let body = json(&format!(r#"{{"state":null,{suffix}}}"#));
            assert!(parse_systemone_state(&body, StateNullability::AllowNull).is_err());
        }
        for source in [
            "null",
            "7",
            "false",
            "[]",
            r#""RAW""#,
            "{}",
            r#"{"data":null,"url":"RAW"}"#,
            r#"{"data":"https://example.invalid/x"}"#,
            r#"{"url":"file:///private/x"}"#,
        ] {
            let body = json(&format!(
                r#"{{"state":[{{"content":[{{"type":"input_audio","input_audio":{source}}}]}}]}}"#
            ));
            assert!(parse_systemone_state(&body, StateNullability::AllowNull)
                .unwrap_err()
                .to_string()
                .contains("audio must be base64 data"));
        }
        Ok(())
    }
}
