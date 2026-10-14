//! Question-only systemone boundary. State/media interpretation and request-byte
//! admission belong to the model-specific front end, not this parser.
use super::{OrderedJson, Question};
use lloom_tensor::{bail, Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NoulAliases {
    TrueFalseOnly,
    TrueFalseAndYesNo,
}

/// Parse already ordered JSON without coercing instructions or descriptions.
/// Missing and null top-level noul criteria both mean no descriptions, as in the
/// reference. An explicitly null description stays present and overrides aliases.
/// Option limits are validated before cloning any question payload or options.
/// State presence/nullability and media decoding remain separate obligations.
pub fn parse_systemone_questions(
    body: &OrderedJson,
    max_options: usize,
    aliases: NoulAliases,
) -> Result<Vec<(String, Question<OrderedJson>)>> {
    let Some(OrderedJson::Object(questions)) = body.get("questions") else {
        bail!("\"questions\" must be a non-empty object")
    };
    if questions.is_empty() {
        bail!("\"questions\" must be a non-empty object")
    }
    let mut out = Vec::with_capacity(questions.len());
    for (id, raw) in questions {
        let error = |message: &str| Error::Msg(format!("questions.{id}: {message}"));
        if !matches!(raw, OrderedJson::Object(_)) {
            return Err(error("must be an object"));
        }
        let instructions = raw
            .get("instructions")
            .filter(|v| !matches!(v, OrderedJson::Null))
            .ok_or_else(|| error("\"instructions\" must be provided"))?;
        let criteria = raw.get("criteria");
        let check_count = |count| -> Result<()> {
            if count > max_options {
                return Err(error(&format!(
                    "too many options ({count}), this model supports at most {max_options}"
                )));
            }
            Ok(())
        };
        let question = match raw.get("type").and_then(OrderedJson::as_str) {
            Some("choice") => {
                let Some(OrderedJson::Object(entries)) = criteria else {
                    return Err(error("\"criteria\" must be a non-empty object"));
                };
                if entries.is_empty() {
                    return Err(error("\"criteria\" must be a non-empty object"));
                }
                check_count(entries.len())?;
                Question::Choice {
                    instructions: instructions.clone(),
                    criteria: entries
                        .iter()
                        .map(|(key, value)| (key.clone(), Some(value.clone())))
                        .collect(),
                }
            }
            Some("score") => {
                let Some(OrderedJson::Array(levels)) = criteria else {
                    return Err(error("\"criteria\" must be an array of 2 to 10 levels"));
                };
                if !(2..=10).contains(&levels.len()) {
                    return Err(error("\"criteria\" must be an array of 2 to 10 levels"));
                }
                check_count(levels.len())?;
                Question::Score {
                    instructions: instructions.clone(),
                    criteria: levels.clone(),
                }
            }
            Some("noul") => {
                if !matches!(
                    criteria,
                    None | Some(OrderedJson::Null | OrderedJson::Object(_))
                ) {
                    return Err(error("\"criteria\" must be an object"));
                }
                check_count(2)?;
                let descriptions = criteria.and_then(|value| {
                    if !matches!(value, OrderedJson::Object(_)) {
                        return None;
                    }
                    let description = |key, alias| {
                        value
                            .get(key)
                            .or_else(|| {
                                (aliases == NoulAliases::TrueFalseAndYesNo)
                                    .then(|| value.get(alias))
                                    .flatten()
                            })
                            .cloned()
                    };
                    Some((description("true", "yes"), description("false", "no")))
                });
                Question::Noul {
                    instructions: instructions.clone(),
                    criteria: descriptions,
                }
            }
            _ => return Err(error("\"type\" must be one of: choice, score, noul")),
        };
        question.validate()?;
        out.push((id.clone(), question));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn parse(
        text: &str,
        limit: usize,
        aliases: NoulAliases,
    ) -> Result<Vec<(String, Question<OrderedJson>)>> {
        parse_systemone_questions(
            &OrderedJson::from_slice(text.as_bytes()).unwrap(),
            limit,
            aliases,
        )
    }
    #[test]
    fn preserves_question_order_raw_values_and_last_duplicate_value() -> Result<()> {
        let questions = parse(
            r#"{"questions":{"z":{"type":"choice","instructions":false,"criteria":{"B":null,"A":{"z":"雪\n","a":[false]},"B":17}},"a":{"type":"score","instructions":{"x":1},"criteria":[null,true]}}}"#,
            3,
            NoulAliases::TrueFalseOnly,
        )?;
        assert_eq!(
            questions
                .iter()
                .map(|(id, _)| id.as_str())
                .collect::<Vec<_>>(),
            ["z", "a"]
        );
        let Question::Choice {
            instructions,
            criteria,
        } = &questions[0].1
        else {
            panic!("choice")
        };
        assert_eq!(instructions, &OrderedJson::Bool(false));
        assert_eq!(
            criteria
                .iter()
                .map(|(id, _)| id.as_str())
                .collect::<Vec<_>>(),
            ["B", "A"]
        );
        assert_eq!(
            serde_json::to_string(criteria[0].1.as_ref().unwrap()).unwrap(),
            "17"
        );
        assert_eq!(
            serde_json::to_string(criteria[1].1.as_ref().unwrap()).unwrap(),
            r#"{"z":"雪\n","a":[false]}"#
        );
        assert!(questions[1]
            .1
            .systemone_answer(&[0.25, 0.75])?
            .to_json_string()
            .unwrap()
            .ends_with(r#""legend":{"0":null,"1":true}}"#));
        Ok(())
    }
    #[test]
    fn noul_alias_policy_preserves_explicit_null_over_alias() -> Result<()> {
        let request = r#"{"questions":{"q":{"type":"noul","instructions":0,"criteria":{"true":null,"yes":"alias yes","no":false}}}}"#;
        for (policy, no) in [
            (NoulAliases::TrueFalseOnly, None),
            (
                NoulAliases::TrueFalseAndYesNo,
                Some(OrderedJson::Bool(false)),
            ),
        ] {
            let questions = parse(request, 2, policy)?;
            let Question::Noul { criteria, .. } = &questions[0].1 else {
                panic!("noul")
            };
            assert_eq!(criteria, &Some((Some(OrderedJson::Null), no)));
        }
        for tail in ["", r#", "criteria":null"#] {
            let questions = parse(
                &format!(r#"{{"questions":{{"q":{{"type":"noul","instructions":"x"{tail}}}}}}}"#),
                2,
                NoulAliases::TrueFalseOnly,
            )?;
            assert!(matches!(
                &questions[0].1,
                Question::Noul { criteria: None, .. }
            ));
        }
        let questions = parse(
            r#"{"questions":{"q":{"type":"noul","instructions":"x","criteria":{}}}}"#,
            2,
            NoulAliases::TrueFalseOnly,
        )?;
        assert!(matches!(
            &questions[0].1,
            Question::Noul {
                criteria: Some((None, None)),
                ..
            }
        ));
        Ok(())
    }
    #[test]
    fn malformed_questions_and_option_limits_reject() {
        for request in [
            r#"{}"#,
            r#"{"questions":[]}"#,
            r#"{"questions":{}}"#,
            r#"{"questions":{"q":null}}"#,
            r#"{"questions":{"q":{"type":"noul"}}}"#,
            r#"{"questions":{"q":{"type":"noul","instructions":null}}}"#,
            r#"{"questions":{"q":{"type":3,"instructions":0}}}"#,
            r#"{"questions":{"q":{"type":"choice","instructions":0,"criteria":{}}}}"#,
            r#"{"questions":{"q":{"type":"score","instructions":0,"criteria":[0]}}}"#,
            r#"{"questions":{"q":{"type":"noul","instructions":0,"criteria":[]}}}"#,
        ] {
            assert!(
                parse(request, 255, NoulAliases::TrueFalseOnly).is_err(),
                "{request}"
            );
        }
        for kind in [
            r#""choice","criteria":{"A":null,"B":null}"#,
            r#""score","criteria":[0,1]"#,
            r#""noul""#,
        ] {
            let request =
                format!(r#"{{"questions":{{"q":{{"instructions":"x","type":{kind}}}}}}}"#);
            let error = parse(&request, 1, NoulAliases::TrueFalseOnly)
                .unwrap_err()
                .to_string();
            assert!(error.contains("too many options (2)"), "{error}");
        }
    }
}
