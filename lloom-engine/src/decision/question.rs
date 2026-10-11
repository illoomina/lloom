use lloom_tensor::{bail, Result};
use serde::ser::SerializeMap;
use serde::{Serialize, Serializer};
use serde_json::{json, Value};

/// A named question's payload. Choice and score order is preserved verbatim.
#[derive(Clone, Debug, PartialEq)]
pub enum Question<Content = String> {
    Noul {
        instructions: Content,
        /// The optional descriptions for true and false, respectively.
        criteria: Option<(Option<Content>, Option<Content>)>,
    },
    Choice {
        instructions: Content,
        criteria: Vec<(String, Option<Content>)>,
    },
    Score {
        instructions: Content,
        criteria: Vec<Content>,
    },
}

impl<Content> Question<Content> {
    pub fn validate(&self) -> Result<()> {
        match self {
            Self::Choice { criteria, .. } => {
                if criteria.is_empty() {
                    bail!("a choice needs at least one option");
                }
                for (i, (name, _)) in criteria.iter().enumerate() {
                    if criteria[..i].iter().any(|(other, _)| other == name) {
                        bail!("duplicate choice option {name:?}");
                    }
                }
            }
            Self::Score { criteria, .. } if !(2..=10).contains(&criteria.len()) => {
                bail!("a score needs 2 to 10 ordered levels");
            }
            _ => {}
        }
        Ok(())
    }

    pub fn option_count(&self) -> usize {
        match self {
            Self::Noul { .. } => 2,
            Self::Choice { criteria, .. } => criteria.len(),
            Self::Score { criteria, .. } => criteria.len(),
        }
    }

    pub fn temperature_key(&self) -> String {
        let kind = match self {
            Self::Noul { .. } => "noul",
            Self::Choice { .. } => "choice",
            Self::Score { .. } => "score",
        };
        let bucket = match self.option_count() {
            0..=2 => "2",
            3..=5 => "3-5",
            6..=10 => "6-10",
            _ => "11+",
        };
        format!("{kind}:{bucket}")
    }

    /// llama.cpp `/v1/systemone` response confidence. Probabilities and selected
    /// answers are unchanged; the Python reference's max-probability confidence
    /// remains available through `answer` for explicit reference comparisons.
    pub fn systemone_answer(&self, probabilities: &[f64]) -> Result<Answer<Content>>
    where
        Content: Clone,
    {
        let mut answer = self.answer(probabilities)?;
        match &mut answer {
            Answer::Noul { .. } => {}
            Answer::Choice { confidence, .. } => {
                let uniform = 1.0 / probabilities.len() as f64;
                *confidence = if probabilities.len() == 1 {
                    1.0
                } else {
                    ((*confidence - uniform) / (1.0 - uniform)).max(0.0)
                };
            }
            Answer::Score { confidence, .. } => {
                let mode = probabilities.iter().enumerate().fold(0, |best, (i, p)| {
                    if *p > probabilities[best] {
                        i
                    } else {
                        best
                    }
                });
                let n = probabilities.len() as f64;
                let distance: f64 = probabilities
                    .iter()
                    .enumerate()
                    .map(|(i, p)| p * (i as f64 - mode as f64).abs())
                    .sum();
                let uniform: f64 = (0..probabilities.len())
                    .map(|i| (i as f64 - (n - 1.0) / 2.0).abs() / n)
                    .sum();
                *confidence = (1.0 - distance / uniform).max(0.0);
            }
        }
        Ok(answer)
    }

    /// Build a response from probabilities in question order; noul is yes, no.
    pub fn answer(&self, probabilities: &[f64]) -> Result<Answer<Content>>
    where
        Content: Clone,
    {
        self.validate()?;
        if probabilities.len() != self.option_count()
            || probabilities
                .iter()
                .any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
            || (probabilities.iter().sum::<f64>() - 1.0).abs() > 1e-6
        {
            bail!(
                "decision probabilities must be a finite normalized distribution over every option"
            );
        }
        let best = probabilities.iter().enumerate().fold(0, |best, (i, p)| {
            if *p > probabilities[best] {
                i
            } else {
                best
            }
        });
        Ok(match self {
            Self::Noul { .. } => Answer::Noul {
                probability: probabilities[0],
            },
            Self::Choice { criteria, .. } => Answer::Choice {
                choice: criteria[best].0.clone(),
                confidence: probabilities[best],
                probabilities: criteria
                    .iter()
                    .zip(probabilities)
                    .map(|((name, _), &p)| (name.clone(), p))
                    .collect(),
            },
            Self::Score { criteria, .. } => Answer::Score {
                score: probabilities
                    .iter()
                    .enumerate()
                    .map(|(i, p)| i as f64 * p)
                    .sum(),
                confidence: probabilities[best],
                probabilities: probabilities.to_vec(),
                legend: criteria.clone(),
            },
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Answer<Content = String> {
    Noul {
        probability: f64,
    },
    Choice {
        choice: String,
        confidence: f64,
        probabilities: Vec<(String, f64)>,
    },
    Score {
        score: f64,
        confidence: f64,
        probabilities: Vec<f64>,
        legend: Vec<Content>,
    },
}

impl<Content: Serialize> Answer<Content> {
    /// Convenience value conversion. Object key order is not retained by the
    /// workspace's serde_json map; use Serialize or to_json_string on the wire.
    pub fn to_json(&self) -> Value {
        match self {
            Self::Noul { probability } => json!({"type": "noul", "noul": probability}),
            Self::Choice {
                choice,
                confidence,
                probabilities,
            } => {
                let probabilities: serde_json::Map<String, Value> = probabilities
                    .iter()
                    .map(|(name, p)| (name.clone(), json!(p)))
                    .collect();
                json!({"type": "choice", "choice": choice, "confidence": confidence,
                    "probabilities": probabilities})
            }
            Self::Score {
                score,
                confidence,
                probabilities,
                legend,
            } => {
                let probabilities: serde_json::Map<String, Value> = probabilities
                    .iter()
                    .enumerate()
                    .map(|(i, p)| (i.to_string(), json!(p)))
                    .collect();
                let legend: serde_json::Map<String, Value> = legend
                    .iter()
                    .enumerate()
                    .map(|(i, text)| (i.to_string(), json!(text)))
                    .collect();
                json!({"type": "score", "score": score, "confidence": confidence,
                    "probabilities": probabilities, "legend": legend})
            }
        }
    }
}

/// Serialize ordered pairs as an object without a sorted-map intermediary.
struct Pairs<'a, T>(&'a [(String, T)]);
impl<T: Serialize> Serialize for Pairs<'_, T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.0.len()))?;
        for (key, value) in self.0 {
            map.serialize_entry(key, value)?;
        }
        map.end()
    }
}

struct Levels<'a, T>(&'a [T]);
impl<T: Serialize> Serialize for Levels<'_, T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.0.len()))?;
        for (index, value) in self.0.iter().enumerate() {
            map.serialize_entry(&index.to_string(), value)?;
        }
        map.end()
    }
}

impl<Content: Serialize> Serialize for Answer<Content> {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(None)?;
        match self {
            Self::Noul { probability } => {
                map.serialize_entry("type", "noul")?;
                map.serialize_entry("noul", probability)?;
            }
            Self::Choice {
                choice,
                confidence,
                probabilities,
            } => {
                map.serialize_entry("type", "choice")?;
                map.serialize_entry("choice", choice)?;
                map.serialize_entry("confidence", confidence)?;
                map.serialize_entry("probabilities", &Pairs(probabilities))?;
            }
            Self::Score {
                score,
                confidence,
                probabilities,
                legend,
            } => {
                map.serialize_entry("type", "score")?;
                map.serialize_entry("score", score)?;
                map.serialize_entry("confidence", confidence)?;
                map.serialize_entry("probabilities", &Levels(probabilities))?;
                map.serialize_entry("legend", &Levels(legend))?;
            }
        }
        map.end()
    }
}

impl<Content: Serialize> Answer<Content> {
    pub fn to_json_string(&self) -> serde_json::Result<String> {
        serde_json::to_string(self)
    }
}

#[cfg(test)]
mod tests {
    use super::Question;
    #[test]
    fn wire_answer_preserves_option_order() {
        let q: Question = Question::Choice {
            instructions: "Pick.".into(),
            criteria: vec![("zeta".into(), None), ("alpha".into(), None)],
        };
        let answer = q.answer(&[0.5, 0.5]).unwrap().to_json_string().unwrap();
        assert_eq!(
            answer,
            r#"{"type":"choice","choice":"zeta","confidence":0.5,"probabilities":{"zeta":0.5,"alpha":0.5}}"#
        );
    }

    #[test]
    fn systemone_confidence_matches_llama_formula_not_python_maximum() {
        let q: Question = Question::Choice {
            instructions: "Pick.".into(),
            criteria: vec![("z".into(), None), ("a".into(), None)],
        };
        assert!(
            (q.systemone_answer(&[0.8, 0.2]).unwrap().to_json()["confidence"]
                .as_f64()
                .unwrap()
                - 0.6)
                .abs()
                < 1e-14
        );
        assert_eq!(
            q.systemone_answer(&[0.5, 0.5]).unwrap().to_json()["confidence"],
            0.0
        );
        assert_eq!(q.answer(&[0.5, 0.5]).unwrap().to_json()["confidence"], 0.5);
        let q: Question = Question::Score {
            instructions: "Rate.".into(),
            criteria: vec!["low".into(), "mid".into(), "high".into()],
        };
        assert!(
            (q.systemone_answer(&[0.2, 0.5, 0.3]).unwrap().to_json()["confidence"]
                .as_f64()
                .unwrap()
                - 0.25)
                .abs()
                < 1e-14
        );
    }
    #[test]
    fn systemone_singleton_choice_is_defined() {
        let q: Question = Question::Choice {
            instructions: "Pick.".into(),
            criteria: vec![("only".into(), None)],
        };
        assert_eq!(
            q.systemone_answer(&[1.0]).unwrap().to_json()["confidence"],
            1.0
        );
    }

    #[test]
    fn raw_json_payload_keeps_nested_score_legend_on_wire() {
        use super::super::OrderedJson;
        let levels = [r#"{"z":"雪\nline","a":[false,null]}"#, "17"]
            .into_iter()
            .map(|s| OrderedJson::from_slice(s.as_bytes()).unwrap())
            .collect();
        let q: Question<OrderedJson> = Question::Score {
            instructions: OrderedJson::Bool(false),
            criteria: levels,
        };
        q.validate().unwrap();
        assert_eq!(q.option_count(), 2);
        assert_eq!(q.temperature_key(), "score:2");
        let answer = q.systemone_answer(&[0.25, 0.75]).unwrap();
        assert_eq!(
            answer.to_json_string().unwrap(),
            r#"{"type":"score","score":0.75,"confidence":0.5,"probabilities":{"0":0.25,"1":0.75},"legend":{"0":{"z":"雪\nline","a":[false,null]},"1":17}}"#
        );
        let super::Answer::Score { legend, .. } = answer else {
            panic!("score answer")
        };
        let Question::Score { criteria, .. } = q else {
            panic!("score question")
        };
        assert_eq!(legend, criteria);
    }

    #[test]
    fn cardinality_and_validation_do_not_require_payload_traits() {
        struct Opaque;
        let q = Question::Choice {
            instructions: Opaque,
            criteria: vec![("z".into(), Some(Opaque)), ("a".into(), None)],
        };
        q.validate().unwrap();
        assert_eq!(q.option_count(), 2);
        assert_eq!(q.temperature_key(), "choice:2");
        let duplicate = Question::Choice {
            instructions: Opaque,
            criteria: vec![("z".into(), None), ("z".into(), None)],
        };
        assert!(duplicate.validate().is_err());
    }
}
