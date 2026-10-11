//! d1-omni question preparation, not causal token-code readout.
//!
//! Pinned reference: LiquidAI/d1-omni-600M
//! 02b55d7076f15129e59ab3f94783f32c4b088674, prompt.py::encode.
//! Media is encoded once outside this component; each question has its own
//! bidirectional text sequence. Marker positions are relative to that text,
//! excluding the media prefix (the head receives the text slice only).

use super::Question;
use lloom_tensor::{bail, Result};

pub(super) const OMNI_SEPARATOR: &str = "<<d1omni:sep>>";
pub(super) const OMNI_STATE: &str = "<<d1omni:state>>";
pub(super) const OMNI_QUESTION: &str = "<<d1omni:question>>";
pub(super) const OMNI_OPTION: &str = "<<d1omni:option>>";

fn option_budget(n: usize, max_len: usize) -> Result<(usize, usize)> {
    if n == 0 {
        bail!("d1-omni needs at least one option")
    }
    let option_budget = n
        .checked_mul(24)
        .and_then(|v| v.checked_add(32))
        .ok_or_else(|| lloom_tensor::Error::msg("d1-omni option budget overflow"))?;
    let framing = n
        .checked_mul(3)
        .ok_or_else(|| lloom_tensor::Error::msg("d1-omni option framing overflow"))?;
    let budget = 96.max(option_budget.min(max_len / 2));
    let per = 2.max(budget.saturating_sub(framing) / n);
    Ok((16.max(budget), per))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OmniMedia {
    Text,
    Image,
    Audio,
}

/// Resolve once from the loaded tokenizer, not hardcoded vocabulary offsets.
#[derive(Clone, Copy, Debug)]
pub struct OmniSpecialTokens {
    pub bos: u32,
    pub state: u32,
    pub question: u32,
    pub option: u32,
    pub marker: u32,
    pub option_end: u32,
    pub decide: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OmniPreparedQuestion {
    pub ids: Vec<u32>,
    pub markers: Vec<usize>,
    /// Checkpoint head type-embedding row: choice=0, score=1, noul=2.
    pub question_type: usize,
}

impl OmniPreparedQuestion {
    /// Tokenize the marked pieces emitted by a named systemone template.
    /// `encode` parses its explicit special tokens and adds no automatic BOS/EOS.
    /// Caller supplies the media-adjusted text allowance and exact repeated media
    /// marker string; no media tensor is read or created here. Returned markers
    /// are relative to TEXT, as DecisionCore's head requires (not media+text).
    ///
    /// Raw endpoint singleton choices are supported; the existing Python
    /// preparation below deliberately retains its >=2 option contract.
    /// Request/token-vector allocation admission remains the caller's duty.
    pub fn from_systemone_rendered<T>(
        rendered: &str,
        question: &Question<T>,
        expected_media_markers: &str,
        marker: u32,
        max_len: usize,
        mut encode: impl FnMut(&str) -> Result<Vec<u32>>,
    ) -> Result<Self> {
        question.validate()?;
        let n = question.option_count();
        let (question_budget, per) = option_budget(n, max_len)?;
        let mut pieces = rendered.split(OMNI_SEPARATOR);
        if pieces.next() != Some(expected_media_markers) {
            bail!("unexpected layout of the decision prompt: media prefix")
        }
        let mut head = Vec::new();
        let mut state = None;
        let mut tail = Vec::new();
        for piece in pieces {
            if let Some(piece) = piece.strip_prefix(OMNI_STATE) {
                if state.is_some() {
                    bail!("unexpected layout of the decision prompt: repeated state")
                }
                state = Some(encode(piece)?);
                continue;
            }
            let (piece, budget) = if let Some(piece) = piece.strip_prefix(OMNI_QUESTION) {
                (piece, Some(question_budget))
            } else if let Some(piece) = piece.strip_prefix(OMNI_OPTION) {
                (piece, Some(per))
            } else {
                (piece, None)
            };
            let mut ids = encode(piece)?;
            if let Some(budget) = budget {
                ids.truncate(budget);
            }
            if state.is_some() {
                tail.extend(ids);
            } else {
                head.extend(ids);
            }
        }
        let Some(mut state) = state else {
            bail!("unexpected layout of the decision prompt: missing state")
        };
        let framing = head
            .len()
            .checked_add(tail.len())
            .ok_or_else(|| lloom_tensor::Error::msg("d1-omni text framing overflow"))?;
        state.truncate(max_len.saturating_sub(framing));
        head.extend(state);
        head.extend(tail);
        head.truncate(max_len);
        let markers: Vec<usize> = head
            .iter()
            .enumerate()
            .filter_map(|(i, &id)| (id == marker).then_some(i))
            .collect();
        if markers.len() != n {
            bail!("the options do not fit in the context")
        }
        let question_type = match question {
            Question::Choice { .. } => 0,
            Question::Score { .. } => 1,
            Question::Noul { .. } => 2,
        };
        Ok(Self {
            ids: head,
            markers,
            question_type,
        })
    }

    /// `state_text` is already serialized by the model-specific frontend:
    /// strings stay literal, JSON uses the omni reference's compact-spaced
    /// representation, not d1-3B's pretty JSON renderer. `max_len` is the
    /// text allowance after accounting for media and the model context bound.
    /// `encode` omits all automatically added special tokens. Caller text is
    /// escaped before it reaches that encoder, preventing forged markers.
    pub fn prepare(
        state_text: &str,
        question: &Question,
        media: OmniMedia,
        tokens: OmniSpecialTokens,
        max_len: usize,
        mut encode: impl FnMut(&str) -> Result<Vec<u32>>,
    ) -> Result<Self> {
        question.validate()?;
        // The model's Python preparation requires two choice options, even
        // though llama's generic causal endpoint also admits singleton choice.
        if question.option_count() < 2 {
            bail!("d1-omni preparation needs at least two options");
        }
        let (instructions, question_type) = match question {
            Question::Choice { instructions, .. } => (instructions, 0),
            Question::Score { instructions, .. } => (instructions, 1),
            Question::Noul { instructions, .. } => (instructions, 2),
        };
        let options = render_options(question, media);
        let n = options.len();
        let (question_budget, per) = option_budget(n, max_len)?;
        let mut suffix = vec![tokens.question];
        suffix.extend(encode(&escape(instructions))?);
        suffix.truncate(question_budget);
        let mut markers = Vec::with_capacity(n);
        for option in options {
            markers.push(suffix.len() + 1);
            suffix.push(tokens.option);
            suffix.push(tokens.marker);
            let mut ids = encode(&escape(&format!(" {option}")))?;
            ids.truncate(per);
            suffix.extend(ids);
            suffix.push(tokens.option_end);
        }
        suffix.push(tokens.decide);
        let room = max_len.saturating_sub(suffix.len().saturating_add(2));
        let mut state = encode(&escape(state_text))?;
        state.truncate(room);
        let marker_offset = state.len() + 2; // BOS and state delimiter.
        for marker in &mut markers {
            *marker += marker_offset;
        }
        if markers.last().is_some_and(|&m| m >= max_len) {
            bail!("the options do not fit in the context");
        }
        let mut ids = vec![tokens.bos, tokens.state];
        ids.extend(state);
        ids.extend(suffix);
        ids.truncate(max_len);
        Ok(Self {
            ids,
            markers,
            question_type,
        })
    }
}

/// Exactly the reference ASCII name class `<|[A-Za-z0-9_]+|>`, not arbitrary
/// angle-bracket syntax or Unicode names. Replacements are literal caller text.
pub(super) fn escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find("<|") {
        out.push_str(&rest[..start]);
        rest = &rest[start + 2..];
        if let Some(end) = rest.find("|>") {
            let name = &rest[..end];
            if !name.is_empty() && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_') {
                out.push_str("<¦");
                out.push_str(name);
                out.push_str("¦>");
                rest = &rest[end + 2..];
                continue;
            }
        }
        out.push_str("<|");
    }
    out.push_str(rest);
    out
}

fn render_options(q: &Question, media: OmniMedia) -> Vec<String> {
    let spoken = media == OmniMedia::Audio;
    match q {
        Question::Choice { criteria, .. } => criteria
            .iter()
            .enumerate()
            .map(|(i, (name, desc))| {
                let desc = desc.as_deref().filter(|s| !s.is_empty());
                if spoken {
                    format!("option_{i:03}: {}", desc.unwrap_or(name))
                } else {
                    desc.map_or_else(|| name.clone(), |d| format!("{name}: {d}"))
                }
            })
            .collect(),
        Question::Score { criteria, .. } => criteria
            .iter()
            .enumerate()
            .map(|(i, text)| format!("level {i}: {text}"))
            .collect(),
        Question::Noul { criteria, .. } => {
            if spoken {
                return vec!["false: no".into(), "true: yes".into()];
            }
            let (yes, no) = match criteria {
                Some((yes, no)) => (yes.as_deref(), no.as_deref()),
                None if media == OmniMedia::Image => (Some("yes"), Some("no")),
                None => (None, None),
            };
            vec![
                format!(
                    "false: {}",
                    no.filter(|s| !s.is_empty())
                        .unwrap_or("no, the statement does not hold")
                ),
                format!(
                    "true: {}",
                    yes.filter(|s| !s.is_empty())
                        .unwrap_or("yes, the statement holds")
                ),
            ]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decision::OrderedJson as OrderedJsonForTest;

    #[test]
    fn systemone_piece_budget_and_text_relative_markers_follow_reference() -> Result<()> {
        let q = Question::Choice {
            instructions: OrderedJsonForTest::Null,
            criteria: vec![("A".into(), None), ("B".into(), None)],
        };
        // Tiny stand-in tokenizer: explicit special pieces are individual ids;
        // ordinary bytes are unchanged. No model/tokenizer file is loaded.
        let encode = |text: &str| -> Result<Vec<u32>> {
            Ok(text
                .bytes()
                .map(|b| match b {
                    b'@' => 1000,
                    b'#' => 1001,
                    b'?' => 1002,
                    b'!' => 1003,
                    _ => u32::from(b),
                })
                .collect())
        };
        let render = format!("MM{OMNI_SEPARATOR}@#{OMNI_SEPARATOR}{OMNI_STATE}abcdefghijklmnopqrst{OMNI_SEPARATOR}{OMNI_QUESTION}?q{OMNI_SEPARATOR}!{OMNI_SEPARATOR}{OMNI_OPTION} A{OMNI_SEPARATOR}!{OMNI_SEPARATOR}{OMNI_OPTION} B");
        let got =
            OmniPreparedQuestion::from_systemone_rendered(&render, &q, "MM", 1003, 16, encode)?;
        assert_eq!(
            got.ids,
            [1000, 1001, 97, 98, 99, 100, 101, 102, 1002, 113, 1003, 32, 65, 1003, 32, 66]
        );
        assert_eq!(got.markers, [10, 13]);
        assert_eq!(got.question_type, 0);
        // Exact pinned budget: n2/max256 =>96 question,45 per option.
        let long = format!("{OMNI_SEPARATOR}@#{OMNI_SEPARATOR}{OMNI_STATE}s{OMNI_SEPARATOR}{OMNI_QUESTION}{}{OMNI_SEPARATOR}!{OMNI_SEPARATOR}{OMNI_OPTION}{}{OMNI_SEPARATOR}!{OMNI_SEPARATOR}{OMNI_OPTION}{}", "q".repeat(120),"a".repeat(60),"b".repeat(60));
        let got = OmniPreparedQuestion::from_systemone_rendered(&long, &q, "", 1003, 256, encode)?;
        assert_eq!(got.ids.len(), 191);
        assert_eq!(got.markers, [99, 145]);
        assert_eq!(
            got.ids.iter().filter(|&&v| v == u32::from(b'q')).count(),
            96
        );
        assert_eq!(
            got.ids.iter().filter(|&&v| v == u32::from(b'a')).count(),
            45
        );
        assert_eq!(
            got.ids.iter().filter(|&&v| v == u32::from(b'b')).count(),
            45
        );
        Ok(())
    }

    #[test]
    fn systemone_layout_and_marker_cardinality_refuse_without_model_work() {
        let q: Question<OrderedJsonForTest> = Question::Choice {
            instructions: OrderedJsonForTest::Null,
            criteria: vec![("A".into(), None)],
        };
        let encode = |s: &str| Ok(s.bytes().map(u32::from).collect());
        for (prompt, media, limit, expected) in [
            (
                format!("wrong{OMNI_SEPARATOR}{OMNI_STATE}!"),
                "",
                8,
                "media prefix",
            ),
            (format!("{OMNI_SEPARATOR}!"), "", 8, "missing state"),
            (
                format!("{OMNI_SEPARATOR}{OMNI_STATE}!{OMNI_SEPARATOR}{OMNI_STATE}!"),
                "",
                8,
                "repeated state",
            ),
            (
                format!("{OMNI_SEPARATOR}{OMNI_STATE}s{OMNI_SEPARATOR}!"),
                "",
                0,
                "options do not fit",
            ),
            (
                format!("{OMNI_SEPARATOR}{OMNI_STATE}s{OMNI_SEPARATOR}!!"),
                "",
                8,
                "options do not fit",
            ),
        ] {
            let error = OmniPreparedQuestion::from_systemone_rendered(
                &prompt,
                &q,
                media,
                u32::from(b'!'),
                limit,
                encode,
            )
            .unwrap_err();
            assert!(error.to_string().contains(expected), "{error}");
        }
        let prompt = format!("{OMNI_SEPARATOR}{OMNI_STATE}s{OMNI_SEPARATOR}!");
        let one = OmniPreparedQuestion::from_systemone_rendered(
            &prompt,
            &q,
            "",
            u32::from(b'!'),
            8,
            encode,
        )
        .unwrap();
        assert_eq!(one.ids, [u32::from(b's'), u32::from(b'!')]);
        assert_eq!(one.markers, [1]);
    }

    #[test]
    fn escape_only_reference_special_names_including_nested_matches() {
        assert_eq!(
            escape("<|mask|> <|reserved_7|> <|雪|> <||> <|a-<|mask|>|>"),
            "<¦mask¦> <¦reserved_7¦> <|雪|> <||> <|a-<¦mask¦>|>"
        );
    }

    #[test]
    fn media_noul_options_match_training_order_and_defaults() {
        let q = Question::Noul {
            instructions: "holds?".into(),
            criteria: None,
        };
        assert_eq!(
            render_options(&q, OmniMedia::Image),
            ["false: no", "true: yes"]
        );
        assert_eq!(
            render_options(&q, OmniMedia::Audio),
            ["false: no", "true: yes"]
        );
        assert_eq!(
            render_options(&q, OmniMedia::Text),
            [
                "false: no, the statement does not hold",
                "true: yes, the statement holds"
            ]
        );
        let explicit = Question::Noul {
            instructions: "holds?".into(),
            criteria: Some((Some("custom yes".into()), Some("custom no".into()))),
        };
        assert_eq!(
            render_options(&explicit, OmniMedia::Audio),
            ["false: no", "true: yes"]
        );
        assert_eq!(
            render_options(&explicit, OmniMedia::Image),
            ["false: custom no", "true: custom yes"]
        );
    }

    #[test]
    fn preparation_matches_pinned_model_free_python_fixtures() {
        let fixtures: serde_json::Value =
            serde_json::from_str(include_str!("fixtures/omni.json")).unwrap();
        let cases = fixtures["cases"].as_array().unwrap();
        assert_eq!(cases.len(), 9);
        let tokens = OmniSpecialTokens {
            bos: 1000,
            state: 1001,
            question: 1002,
            option: 1003,
            marker: 1004,
            option_end: 1005,
            decide: 1006,
        };
        let mut errors = 0;
        for case in cases {
            let name = case["name"].as_str().unwrap();
            let instructions = case["instructions"].as_str().unwrap().to_owned();
            let criteria = &case["criteria"];
            let question = match case["kind"].as_str().unwrap() {
                "choice" => Question::Choice {
                    instructions,
                    criteria: criteria
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|pair| {
                            (
                                pair[0].as_str().unwrap().to_owned(),
                                pair[1].as_str().map(str::to_owned),
                            )
                        })
                        .collect(),
                },
                "score" => Question::Score {
                    instructions,
                    criteria: criteria
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_str().unwrap().to_owned())
                        .collect(),
                },
                "noul" => Question::Noul {
                    instructions,
                    criteria: criteria.as_array().map(|pairs| {
                        let lookup = |key: &str| {
                            pairs
                                .iter()
                                .find(|pair| pair[0].as_str() == Some(key))
                                .and_then(|pair| pair[1].as_str())
                                .map(str::to_owned)
                        };
                        (lookup("true"), lookup("false"))
                    }),
                },
                other => panic!("unrecognized fixture kind {other}"),
            };
            let media = match case["media"].as_str().unwrap() {
                "text" => OmniMedia::Text,
                "image" => OmniMedia::Image,
                "audio" => OmniMedia::Audio,
                other => panic!("unrecognized fixture media {other}"),
            };
            let got = OmniPreparedQuestion::prepare(
                case["state_text"].as_str().unwrap(),
                &question,
                media,
                tokens,
                case["max_len"].as_u64().unwrap() as usize,
                |text| Ok(text.bytes().map(u32::from).collect()),
            );
            if let Some(error) = case["error"].as_str() {
                assert!(got.unwrap_err().to_string().contains(error), "{name}");
                errors += 1;
            } else {
                let got = got.unwrap();
                let ids: Vec<u32> = serde_json::from_value(case["ids"].clone()).unwrap();
                let markers: Vec<usize> = serde_json::from_value(case["markers"].clone()).unwrap();
                assert_eq!(got.ids, ids, "{name} token sequence");
                assert_eq!(got.markers, markers, "{name} marker positions");
                for &marker in &got.markers {
                    assert_eq!(got.ids[marker], tokens.marker, "{name}");
                }
            }
        }
        assert_eq!(errors, 1, "fixture corpus must exercise marker overflow");
    }
}
