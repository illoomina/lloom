//! Ordered endpoint response, shared by causal and non-generating executors.
//! Wire contract: pinned llama.cpp test_systemone.py (answers + usage, zero
//! generated tokens). Input tokens count logical prompts, including reused ones.
use super::{question::Pairs, Answer, OrderedJson};
use serde::{ser::SerializeStruct, Serialize, Serializer};

#[derive(Debug, thiserror::Error)]
pub enum SystemOneError {
    #[error("{0}")]
    InvalidRequest(String),
    #[error("{0}")]
    Unsupported(String),
    #[error("{0}")]
    Execution(String),
}

/// Answers retain request order; nested legends retain their original JSON.
/// This type serializes directly, without a serde_json::Value intermediary.
#[derive(Debug, Clone)]
pub struct SystemOneResponse {
    answers: Vec<(String, Answer<OrderedJson>)>,
    input_tokens: u64,
}
impl SystemOneResponse {
    pub fn new(
        answers: Vec<(String, Answer<OrderedJson>)>,
        input_tokens: u64,
    ) -> Result<Self, SystemOneError> {
        if answers.is_empty() {
            return Err(SystemOneError::Execution(
                "decision executor returned no answers".into(),
            ));
        }
        for (i, (id, _)) in answers.iter().enumerate() {
            if answers[..i].iter().any(|(other, _)| other == id) {
                return Err(SystemOneError::Execution(format!(
                    "decision executor returned duplicate answer {id:?}"
                )));
            }
        }
        Ok(Self {
            answers,
            input_tokens,
        })
    }
    pub fn answers(&self) -> &[(String, Answer<OrderedJson>)] {
        &self.answers
    }
    pub fn input_tokens(&self) -> u64 {
        self.input_tokens
    }
}
impl Serialize for SystemOneResponse {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        struct Usage {
            input_tokens: u64,
            output_tokens: u64,
        }
        let mut out = serializer.serialize_struct("SystemOneResponse", 2)?;
        out.serialize_field("answers", &Pairs(&self.answers))?;
        out.serialize_field(
            "usage",
            &Usage {
                input_tokens: self.input_tokens,
                output_tokens: 0,
            },
        )?;
        out.end()
    }
}
