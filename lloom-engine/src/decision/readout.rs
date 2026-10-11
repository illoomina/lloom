use super::{Question, Temperature};
use lloom_tensor::{bail, Result};

/// Distinct answer-token aliases, resolved before any model work is submitted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CausalReadout {
    groups: Vec<Vec<u32>>,
}

/// A rendered question plus its readout: both use the same resolved option codes.
#[derive(Clone, Debug)]
pub struct PreparedQuestion {
    pub question: Question,
    pub suffix: String,
    pub readout: CausalReadout,
}

impl PreparedQuestion {
    /// `encode` must omit BOS/EOS and other automatically inserted special tokens.
    pub fn causal(
        question: Question,
        encode: impl FnMut(&str) -> Result<Vec<u32>>,
    ) -> Result<Self> {
        Self::causal_with_labels(question, encode, false)
    }

    /// Label selection of llama.cpp's `/v1/systemone`: native labels are
    /// untrimmed ASCII letters, not Python's stripped Unicode letters.
    pub fn systemone_causal(
        question: Question,
        encode: impl FnMut(&str) -> Result<Vec<u32>>,
    ) -> Result<Self> {
        Self::causal_with_labels(question, encode, true)
    }

    fn causal_with_labels(
        question: Question,
        mut encode: impl FnMut(&str) -> Result<Vec<u32>>,
        systemone: bool,
    ) -> Result<Self> {
        question.validate()?;
        let (body, groups) = match &question {
            Question::Noul {
                instructions,
                criteria,
            } => {
                let yes = single_ids(&["yes", "Yes", "YES"], &mut encode)?;
                let no = single_ids(&["no", "No", "NO"], &mut encode)?;
                if yes.is_empty() || no.is_empty() {
                    bail!("tokenizer has no single-token yes/no aliases");
                }
                let extra = match criteria {
                    None => String::new(),
                    Some((yes, no)) => format!(
                        "\nYes: {}\nNo: {}",
                        yes.as_deref().unwrap_or("None"),
                        no.as_deref().unwrap_or("None")
                    ),
                };
                (
                    format!("{instructions}{extra}\n\nReply with yes or no only."),
                    vec![yes, no],
                )
            }
            Question::Choice {
                instructions,
                criteria,
            } => {
                let labels: Vec<&str> = criteria
                    .iter()
                    .map(|(name, _)| {
                        if systemone {
                            name.as_str()
                        } else {
                            name.trim()
                        }
                    })
                    .collect();
                let native = labels.iter().all(|s| {
                    if systemone {
                        s.len() == 1 && s.as_bytes()[0].is_ascii_alphabetic()
                    } else {
                        let mut chars = s.chars();
                        chars.next().is_some_and(char::is_alphabetic) && chars.next().is_none()
                    }
                });
                let mut used = Vec::with_capacity(labels.len());
                let mut groups = Vec::with_capacity(labels.len());
                let mut lines = Vec::with_capacity(labels.len());
                for (i, (label, desc)) in criteria.iter().enumerate() {
                    let requested = if native {
                        labels[i].to_string()
                    } else if labels.len() <= 26 {
                        ((b'A' + i as u8) as char).to_string()
                    } else {
                        format!("{i:02}")
                    };
                    let (code, id) = match available(&requested, &used, &mut encode)? {
                        Some(id) => (requested, id),
                        None => fallback(&used, &mut encode)?,
                    };
                    used.push(id);
                    let mut group = vec![id];
                    let spaced = format!(" {code}");
                    for other in single_ids(&[&spaced], &mut encode)? {
                        if !group.contains(&other) {
                            group.push(other);
                        }
                    }
                    groups.push(group);
                    let text = desc
                        .as_ref()
                        .filter(|s| !s.is_empty())
                        .cloned()
                        .unwrap_or_else(|| label.replace('_', " "));
                    lines.push(format!("{code} {text}"));
                }
                (
                    format!(
                        "{instructions}\n\nOptions:\n{}\n\nReply with the option code only.",
                        lines.join("\n")
                    ),
                    groups,
                )
            }
            Question::Score {
                instructions,
                criteria,
            } => {
                let mut groups = Vec::with_capacity(criteria.len());
                let mut lines = Vec::with_capacity(criteria.len());
                for (i, description) in criteria.iter().enumerate() {
                    let code = i.to_string();
                    let group = single_ids(&[&code], &mut encode)?;
                    if group.is_empty() {
                        bail!("score level {i} is not a single token");
                    }
                    groups.push(group);
                    lines.push(format!("{i} {description}"));
                }
                (
                    format!(
                        "{instructions}\n\n{}\n\nReply with a single digit 0-{} only.",
                        lines.join("\n"),
                        criteria.len() - 1
                    ),
                    groups,
                )
            }
        };
        Ok(Self {
            question,
            suffix: format!("{body}<|im_end|>\n<|im_start|>assistant\n"),
            readout: CausalReadout { groups },
        })
    }
}

impl CausalReadout {
    pub fn groups(&self) -> &[Vec<u32>] {
        &self.groups
    }

    /// Max-pool aliases, then softmax options. A vocabulary-wide log-softmax is
    /// unnecessary: its shared additive constant cancels in this normalization.
    /// `logit` permits selected-token readback instead of a full vocabulary copy.
    pub fn probabilities(
        &self,
        mut logit: impl FnMut(u32) -> Result<f64>,
        temperature: f64,
    ) -> Result<Vec<f64>> {
        let temperature = Temperature::try_from(temperature)?;
        let mut scores = Vec::with_capacity(self.groups.len());
        for group in &self.groups {
            let mut score = f64::NEG_INFINITY;
            for &id in group {
                let value = logit(id)?;
                if !value.is_finite() {
                    bail!("non-finite decision logit at token {id}");
                }
                score = score.max(value);
            }
            scores.push(score);
        }
        normalize_options(scores, temperature)
    }
}

/// Shared selected-option normalization; never normalizes padded marker slots.
/// Host F64 arithmetic is intentional; reference F32 parity uses a tolerance.
pub(super) fn normalize_options(
    mut scores: Vec<f64>,
    temperature: Temperature,
) -> Result<Vec<f64>> {
    if scores.is_empty() || scores.iter().any(|score| !score.is_finite()) {
        bail!("decision option logits must be nonempty and finite");
    }
    let temperature = f64::from(temperature);
    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut sum = 0.0;
    for score in &mut scores {
        *score = ((*score - max) / temperature).exp();
        sum += *score;
    }
    for score in &mut scores {
        *score /= sum;
    }
    Ok(scores)
}

fn single_ids(
    texts: &[&str],
    encode: &mut impl FnMut(&str) -> Result<Vec<u32>>,
) -> Result<Vec<u32>> {
    let mut ids = Vec::with_capacity(texts.len());
    for text in texts {
        let tokens = encode(text)?;
        if let [id] = tokens.as_slice() {
            if !ids.contains(id) {
                ids.push(*id);
            }
        }
    }
    Ok(ids)
}

fn available(
    text: &str,
    used: &[u32],
    encode: &mut impl FnMut(&str) -> Result<Vec<u32>>,
) -> Result<Option<u32>> {
    let ids = encode(text)?;
    Ok(match ids.as_slice() {
        [id] if !used.contains(id) => Some(*id),
        _ => None,
    })
}

fn fallback(
    used: &[u32],
    encode: &mut impl FnMut(&str) -> Result<Vec<u32>>,
) -> Result<(String, u32)> {
    let candidates = (b'A'..=b'Z')
        .map(|c| (c as char).to_string())
        .chain((0..100).map(|i| format!("{i:02}")))
        .chain((b'a'..=b'z').map(|c| (c as char).to_string()))
        .chain((0..200).map(|i| format!("#{i}")))
        .chain(
            (b'A'..=b'Z')
                .flat_map(|a| (b'A'..=b'Z').map(move |b| format!("{}{}", a as char, b as char))),
        );
    for text in candidates {
        if let Some(id) = available(&text, used, encode)? {
            return Ok((text, id));
        }
    }
    bail!("tokenizer has no distinct single-token option code remaining")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn choice(names: &[&str]) -> Question {
        Question::Choice {
            instructions: "Pick.".into(),
            criteria: names.iter().map(|s| (s.to_string(), None)).collect(),
        }
    }

    fn bytes(text: &str) -> Result<Vec<u32>> {
        Ok(text.bytes().map(u32::from).collect())
    }

    #[test]
    fn rendered_codes_and_readout_share_fallback_resolution() {
        let prepared = PreparedQuestion::causal(choice(&["multi", "B", "third"]), |s| {
            if s == "A" {
                Ok(vec![8, 9])
            } else {
                bytes(s)
            }
        })
        .unwrap();
        assert!(prepared
            .suffix
            .contains("Options:\nB multi\nC B\nD third\n"));
        assert_eq!(prepared.readout.groups(), &[vec![66], vec![67], vec![68]]);
    }

    #[test]
    fn native_letters_preserve_order_and_spaced_forms_are_max_pooled() {
        let prepared = PreparedQuestion::causal(choice(&["Z", "A"]), |s| match s {
            " Z" => Ok(vec![100]),
            " A" => Ok(vec![101]),
            _ => bytes(s),
        })
        .unwrap();
        assert_eq!(prepared.readout.groups(), &[vec![90, 100], vec![65, 101]]);
        let p = prepared
            .readout
            .probabilities(
                |id| {
                    Ok(match id {
                        100 => 2.0,
                        65 => 1.0,
                        _ => -20.0,
                    })
                },
                1.0,
            )
            .unwrap();
        assert!((p[0] - 0.7310585786300049).abs() < 1e-14);
        assert_eq!(
            prepared.question.answer(&p).unwrap().to_json()["choice"],
            "Z"
        );
    }

    #[test]
    fn score_is_expected_level_not_the_winning_index() {
        let q: Question = Question::Score {
            instructions: "Rate.".into(),
            criteria: vec!["low".into(), "mid".into(), "high".into()],
        };
        let answer = q.answer(&[0.2, 0.5, 0.3]).unwrap().to_json();
        assert!((answer["score"].as_f64().unwrap() - 1.1).abs() < 1e-14);
        assert_eq!(answer["confidence"], 0.5);
        assert_eq!(answer["legend"]["2"], "high");
    }

    #[test]
    fn ties_choose_first_declared_option_not_lexical_order() {
        assert_eq!(
            choice(&["zeta", "alpha"])
                .answer(&[0.5, 0.5])
                .unwrap()
                .to_json()["choice"],
            "zeta"
        );
    }

    #[test]
    fn noul_is_yes_probability_even_when_no_wins() {
        let q = Question::Noul {
            instructions: "True?".into(),
            criteria: None,
        };
        let p = PreparedQuestion::causal(q, |s| {
            Ok(match s {
                "yes" => vec![1],
                "Yes" => vec![2],
                "YES" => vec![3],
                "no" => vec![4],
                "No" => vec![5],
                "NO" => vec![6],
                _ => vec![],
            })
        })
        .unwrap();
        let probs = p
            .readout
            .probabilities(|id| Ok(if id == 5 { 1.0 } else { 0.0 }), 1.0)
            .unwrap();
        assert!((probs[0] - 0.2689414213699951).abs() < 1e-14);
        assert_eq!(
            p.question.answer(&probs).unwrap().to_json()["noul"],
            probs[0]
        );
    }

    #[test]
    fn invalid_inputs_fail_before_forward() {
        assert!(PreparedQuestion::causal(choice(&["same", "same"]), bytes).is_err());
        let p = PreparedQuestion::causal(choice(&["one", "two"]), bytes).unwrap();
        for temp in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(p.readout.probabilities(|_| Ok(0.0), temp).is_err());
        }
        assert!(p.readout.probabilities(|_| Ok(f64::NAN), 1.0).is_err());
        assert!(p.question.answer(&[f64::NAN, 1.0]).is_err());
        assert!(p.question.answer(&[0.3, 0.3]).is_err());
    }

    #[test]
    fn endpoint_labels_do_not_inherit_python_unicode_or_trimming() {
        let encode = |text: &str| Ok(text.chars().map(u32::from).collect());
        let q = choice(&["雪", "α"]);
        let python = PreparedQuestion::causal(q.clone(), encode).unwrap();
        let endpoint = PreparedQuestion::systemone_causal(q, encode).unwrap();
        assert_eq!(python.readout.groups(), &[vec![38634], vec![945]]);
        assert_eq!(endpoint.readout.groups(), &[vec![65], vec![66]]);
        let endpoint = PreparedQuestion::systemone_causal(choice(&[" Z ", "A"]), encode).unwrap();
        assert_eq!(endpoint.readout.groups(), &[vec![65], vec![66]]);
    }
}
