//! The generic decision read for a plain causal model: vLLM's
//! `NextTokenStrategy` (`vllm/entrypoints/generate/structured_decisions`,
//! `2534a96f`). Each question is the state and the question with one label
//! per option (`A`–`Z`, `yes`/`no`, `0`–`9`) as a user turn of the model's own
//! chat template with thinking off; the read is the reply's first token.
//! Uncalibrated: every answer carries the labels' share of the whole
//! vocabulary and whether the most likely token was a label at all.
use super::readout::normalize_options;
use super::{OrderedJson, Question, SystemOneResponse, Temperature};
use lloom_tensor::{bail, Error, Result};
use serde::Serialize;

const CHOICE: [&str; 26] = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S",
    "T", "U", "V", "W", "X", "Y", "Z",
];
const NOUL: [&str; 2] = ["yes", "no"];
const SCORE: [&str; 10] = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];

/// The most options a generic question can carry: one capital letter each.
pub const MAX_OPTIONS: usize = CHOICE.len();

/// Each label's token as the first token of the reply, checked once at load.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GenericLabels {
    tail: Vec<u32>,
    choice: Vec<u32>,
    noul: Vec<u32>,
    score: Vec<u32>,
}

impl GenericLabels {
    /// `text` is a chat prompt rendered to its generation prompt with
    /// thinking off, `prompt` its ids, and `tail` the ids every such prompt
    /// ends with (its generation prompt). Each label, appended to `text`, must
    /// encode as `prompt` plus one token, distinct from its set's other labels
    /// — else the model cannot answer this way and the error names the label.
    pub fn probe(
        text: &str,
        prompt: &[u32],
        tail: Vec<u32>,
        encode: impl Fn(&str) -> Result<Vec<u32>>,
    ) -> Result<Self> {
        if tail.is_empty() || !prompt.ends_with(&tail) {
            bail!("the chat prompt has no stable generation prompt to read a reply after")
        }
        let set = |labels: &[&str]| -> Result<Vec<u32>> {
            let mut ids = Vec::with_capacity(labels.len());
            for label in labels {
                let extended = encode(&format!("{text}{label}"))?;
                match extended.split_last() {
                    Some((&id, head)) if head == prompt && !ids.contains(&id) => ids.push(id),
                    _ => bail!(
                        "decision label {label:?} is not one distinct token after this model's chat prompt"
                    ),
                }
            }
            Ok(ids)
        };
        Ok(Self {
            choice: set(&CHOICE)?,
            noul: set(&NOUL)?,
            score: set(&SCORE)?,
            tail,
        })
    }

    /// The ids every rendered question must end with.
    pub fn tail(&self) -> &[u32] {
        &self.tail
    }

    /// The label tokens of `question`'s options, in option order.
    pub fn ids<T>(&self, question: &Question<T>) -> Result<&[u32]> {
        let count = question.option_count();
        let set = match question {
            Question::Choice { .. } => &self.choice,
            Question::Noul { .. } => &self.noul,
            Question::Score { .. } => &self.score,
        };
        set.get(..count).ok_or_else(|| {
            Error::Msg(format!(
                "the generic decision template labels at most {} options of this type",
                set.len()
            ))
        })
    }
}

/// A JSON value as the question reads it: a string as-is, anything else as
/// compact JSON with Python's separators (`json.dumps`, non-ASCII kept).
pub fn value_text(value: &OrderedJson) -> Result<String> {
    if let OrderedJson::String(text) = value {
        return Ok(text.clone());
    }
    struct Python;
    impl serde_json::ser::Formatter for Python {
        fn begin_array_value<W: ?Sized + std::io::Write>(
            &mut self,
            writer: &mut W,
            first: bool,
        ) -> std::io::Result<()> {
            if first {
                Ok(())
            } else {
                writer.write_all(b", ")
            }
        }
        fn begin_object_key<W: ?Sized + std::io::Write>(
            &mut self,
            writer: &mut W,
            first: bool,
        ) -> std::io::Result<()> {
            if first {
                Ok(())
            } else {
                writer.write_all(b", ")
            }
        }
        fn begin_object_value<W: ?Sized + std::io::Write>(
            &mut self,
            writer: &mut W,
        ) -> std::io::Result<()> {
            writer.write_all(b": ")
        }
    }
    let mut out = Vec::new();
    value
        .serialize(&mut serde_json::Serializer::with_formatter(
            &mut out, Python,
        ))
        .map_err(Error::wrap)?;
    String::from_utf8(out).map_err(Error::wrap)
}

/// The question as the model reads it after the state, with `labels` in
/// option order (vLLM's `QuestionType.prompt`).
pub fn question_text(question: &Question<OrderedJson>) -> Result<String> {
    let mut lines = Vec::with_capacity(question.option_count() + 2);
    let described =
        |label: &str, name: &str, description: Option<&OrderedJson>| -> Result<String> {
            Ok(match description {
                None | Some(OrderedJson::Null) => format!("{label}: {name}"),
                Some(description) => format!("{label}: {name} - {}", value_text(description)?),
            })
        };
    let (instructions, reply) = match question {
        Question::Choice {
            instructions,
            criteria,
        } => {
            for ((name, description), label) in criteria.iter().zip(CHOICE) {
                lines.push(described(label, name, description.as_ref())?);
            }
            (instructions, "Answer with the letter of one option only.")
        }
        Question::Noul {
            instructions,
            criteria,
        } => {
            let (yes, no) = criteria
                .as_ref()
                .map_or((None, None), |(yes, no)| (yes.as_ref(), no.as_ref()));
            for (label, description) in NOUL.into_iter().zip([yes, no]) {
                // A side without a description still shows its label.
                lines.push(match description {
                    None | Some(OrderedJson::Null) => label.to_string(),
                    Some(description) => format!("{label}: {}", value_text(description)?),
                });
            }
            (instructions, "Answer with yes or no only.")
        }
        Question::Score {
            instructions,
            criteria,
        } => {
            for (level, label) in criteria.iter().zip(SCORE) {
                lines.push(format!("{label}: {}", value_text(level)?));
            }
            (instructions, "Answer with the number of one level only.")
        }
    };
    let instructions = value_text(instructions)?;
    if !instructions.is_empty() {
        lines.insert(0, format!("Question: {instructions}"));
    }
    lines.push(reply.to_string());
    Ok(lines.join("\n"))
}

/// The user turn: the state, then the question.
pub fn user_text(state: &OrderedJson, question: &Question<OrderedJson>) -> Result<String> {
    Ok(format!(
        "{}\n\n{}",
        value_text(state)?,
        question_text(question)?
    ))
}

/// One question's read: each label's full-vocabulary log-probability as the
/// reply's first token, in option order, and the most likely token.
pub struct LabelRead {
    pub logprobs: Vec<f32>,
    pub argmax: u32,
    pub input_tokens: u64,
}

/// Per-question signal that the uncalibrated read is or is not about labels.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Diagnostics {
    /// The labels' total probability over the whole vocabulary.
    pub label_mass: f64,
    pub argmax_is_label: bool,
}

/// The answers, in question order, from each question's read.
pub fn answers(
    questions: &[(String, Question<OrderedJson>)],
    reads: &[LabelRead],
    labels: &GenericLabels,
) -> Result<SystemOneResponse> {
    if questions.len() != reads.len() {
        bail!("generic decision read count differs from its questions")
    }
    let mut answers = Vec::with_capacity(questions.len());
    let mut diagnostics = Vec::with_capacity(questions.len());
    let mut input_tokens = 0u64;
    for ((id, question), read) in questions.iter().zip(reads) {
        let ids = labels.ids(question)?;
        if read.logprobs.len() != ids.len() {
            bail!("generic decision read misses a label")
        }
        let scores = read.logprobs.iter().map(|&lp| f64::from(lp)).collect();
        let probabilities = normalize_options(scores, Temperature::try_from(1.)?)?;
        answers.push((id.clone(), question.systemone_answer(&probabilities)?));
        diagnostics.push((
            id.clone(),
            Diagnostics {
                label_mass: read.logprobs.iter().map(|&lp| f64::from(lp).exp()).sum(),
                argmax_is_label: ids.contains(&read.argmax),
            },
        ));
        input_tokens = input_tokens
            .checked_add(read.input_tokens)
            .ok_or_else(|| Error::Msg("decision usage count overflow".into()))?;
    }
    Ok(SystemOneResponse::new(answers, input_tokens)
        .map_err(Error::wrap)?
        .with_diagnostics(diagnostics))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn json(text: &str) -> OrderedJson {
        OrderedJson::from_slice(text.as_bytes()).unwrap()
    }

    /// A toy vocabulary: whole words and single characters are tokens, the
    /// prompt's added markers are ids 900+.
    fn encode(text: &str) -> Result<Vec<u32>> {
        let mut ids = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            if let Some(tail) = rest.strip_prefix("<|im|>") {
                ids.push(900);
                rest = tail;
            } else if let Some(tail) = rest.strip_prefix("yes") {
                ids.push(500);
                rest = tail;
            } else if let Some(tail) = rest.strip_prefix("no") {
                ids.push(501);
                rest = tail;
            } else {
                let c = rest.chars().next().unwrap();
                ids.push(c as u32);
                rest = &rest[c.len_utf8()..];
            }
        }
        Ok(ids)
    }
    #[test]
    fn labels_follow_the_prompt_and_must_be_single_distinct_tokens() -> Result<()> {
        let text = "<|im|>user\nx<|im|>assistant\n";
        let prompt = encode(text)?;
        let tail = encode("<|im|>assistant\n")?;
        let labels = GenericLabels::probe(text, &prompt, tail.clone(), encode)?;
        assert_eq!(labels.tail(), tail.as_slice());
        assert_eq!(labels.noul, vec![500, 501]);
        assert_eq!(labels.choice[0], 'A' as u32);
        assert_eq!(labels.score[9], '9' as u32);
        // A tokenizer that merges the label into the prompt's last token
        // cannot read it as the reply's first token.
        let merging = |text: &str| -> Result<Vec<u32>> {
            let mut ids = encode(text)?;
            if text.ends_with("\nA") {
                ids.pop();
                *ids.last_mut().unwrap() = 7777;
            }
            Ok(ids)
        };
        let error = GenericLabels::probe(text, &prompt, tail.clone(), merging).unwrap_err();
        assert!(error.to_string().contains("\"A\""), "{error}");
        // A label set needs distinct tokens: "yes" and "no" read as one id.
        let collapsing = |text: &str| -> Result<Vec<u32>> {
            let mut ids = encode(text)?;
            if text.ends_with("no") {
                *ids.last_mut().unwrap() = 500;
            }
            Ok(ids)
        };
        let error = GenericLabels::probe(text, &prompt, tail, collapsing).unwrap_err();
        assert!(error.to_string().contains("\"no\""), "{error}");
        Ok(())
    }

    #[test]
    fn question_text_matches_the_reference_shape() -> Result<()> {
        let choice = Question::Choice {
            instructions: json(r#""Which?""#),
            criteria: vec![
                ("red".into(), Some(json(r#""warm""#))),
                ("blue".into(), Some(OrderedJson::Null)),
            ],
        };
        assert_eq!(
            question_text(&choice)?,
            "Question: Which?\nA: red - warm\nB: blue\nAnswer with the letter of one option only."
        );
        let noul = Question::Noul {
            instructions: json(r#""Holds?""#),
            criteria: Some((Some(json(r#""it does""#)), None)),
        };
        assert_eq!(
            question_text(&noul)?,
            "Question: Holds?\nyes: it does\nno\nAnswer with yes or no only."
        );
        let score = Question::Score {
            instructions: json(r#"{"rate":"it","n":1}"#),
            criteria: vec![json(r#""low""#), json(r#""high""#)],
        };
        assert_eq!(
            question_text(&score)?,
            "Question: {\"rate\": \"it\", \"n\": 1}\n0: low\n1: high\nAnswer with the number of one level only."
        );
        assert_eq!(
            user_text(&json(r#"{"a":[1,"é"]}"#), &noul)?,
            "{\"a\": [1, \"é\"]}\n\nQuestion: Holds?\nyes: it does\nno\nAnswer with yes or no only."
        );
        Ok(())
    }

    #[test]
    fn answers_carry_label_mass_and_whether_the_argmax_was_a_label() -> Result<()> {
        let text = "<|im|>assistant\n";
        let prompt = encode(text)?;
        let labels = GenericLabels::probe(text, &prompt, prompt.clone(), encode)?;
        let questions = vec![(
            "q".to_string(),
            Question::Noul {
                instructions: json(r#""?""#),
                criteria: None,
            },
        )];
        let reads = [LabelRead {
            logprobs: vec![0.6f32.ln(), 0.2f32.ln()],
            argmax: 500,
            input_tokens: 7,
        }];
        let response = answers(&questions, &reads, &labels)?;
        let text = serde_json::to_string(&response).map_err(Error::wrap)?;
        let value: serde_json::Value = serde_json::from_str(&text).map_err(Error::wrap)?;
        assert!((value["answers"]["q"]["noul"].as_f64().unwrap() - 0.75).abs() < 1e-6);
        assert!((value["diagnostics"]["q"]["label_mass"].as_f64().unwrap() - 0.8).abs() < 1e-6);
        assert_eq!(value["diagnostics"]["q"]["argmax_is_label"], true);
        assert_eq!(value["usage"]["input_tokens"], 7);
        let off_label = [LabelRead {
            logprobs: vec![0.01f32.ln(), 0.01f32.ln()],
            argmax: 'Q' as u32,
            input_tokens: 7,
        }];
        let text = serde_json::to_string(&answers(&questions, &off_label, &labels)?)
            .map_err(Error::wrap)?;
        assert!(text.contains("\"argmax_is_label\":false"), "{text}");
        Ok(())
    }
}
