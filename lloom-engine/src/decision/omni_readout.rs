use super::{readout::normalize_options, OmniMedia, Question, Temperature};
use lloom_tensor::{bail, Result};
use std::collections::BTreeMap;

/// Calibrated marker-logit readout, prepared before model execution.
/// Omni head order is false/true for noul; public probabilities are yes/no.
#[derive(Clone, Debug)]
pub struct OmniReadout {
    options: usize,
    reverse: bool,
    temperature: Temperature,
}

impl OmniReadout {
    /// Text uses cardinality-specific, then type-specific, then unit temperature.
    /// Image and audio heads bypass checkpoint calibration, as trained.
    pub fn prepare(
        question: &Question,
        media: OmniMedia,
        temperatures: &BTreeMap<String, Temperature>,
    ) -> Result<Self> {
        question.validate()?;
        let options = question.option_count();
        if options < 2 {
            bail!("d1-omni readout needs at least two options");
        }
        let kind = match question {
            Question::Choice { .. } => "choice",
            Question::Score { .. } => "score",
            Question::Noul { .. } => "noul",
        };
        let temperature = if media == OmniMedia::Text {
            temperatures
                .get(&question.temperature_key())
                .or_else(|| temperatures.get(kind))
                .copied()
        } else {
            None
        }
        .unwrap_or(Temperature::try_from(1.)?);
        Ok(Self {
            options,
            reverse: matches!(question, Question::Noul { .. }),
            temperature,
        })
    }

    /// Endpoint readout keeps raw JSON payloads and admits singleton choices.
    /// The Python-reference prepare helper above retains its >=2 contract.
    /// Checkpoint values are borrowed, not copied into a second calibration map.
    pub fn prepare_systemone<T>(
        question: &Question<T>,
        media: OmniMedia,
        temperatures: &BTreeMap<String, f64>,
    ) -> Result<Self> {
        question.validate()?;
        let kind = match question {
            Question::Choice { .. } => "choice",
            Question::Score { .. } => "score",
            Question::Noul { .. } => "noul",
        };
        let temperature = if media == OmniMedia::Text {
            temperatures
                .get(&question.temperature_key())
                .or_else(|| temperatures.get(kind))
                .copied()
        } else {
            None
        }
        .unwrap_or(1.);
        Ok(Self {
            options: question.option_count(),
            reverse: matches!(question, Question::Noul { .. }),
            temperature: Temperature::try_from(temperature)?,
        })
    }

    /// Only the real options are normalized. Trailing padded marker slots are
    /// ignored even when their sentinel is not finite. No generated tokens or
    /// whole-vocabulary readback are needed for this head.
    pub fn probabilities(&self, marker_logits: &[f32]) -> Result<Vec<f64>> {
        if marker_logits.len() < self.options {
            bail!("omni readout is missing marker logits");
        }
        let scores = marker_logits[..self.options]
            .iter()
            .map(|&x| f64::from(x))
            .collect();
        let mut probabilities = normalize_options(scores, self.temperature)?;
        if self.reverse {
            probabilities.reverse();
        }
        Ok(probabilities)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn choice(count: usize) -> Question {
        Question::Choice {
            instructions: "Pick".into(),
            criteria: (0..count).map(|i| (i.to_string(), None)).collect(),
        }
    }

    #[test]
    fn text_calibration_uses_cardinality_then_type_then_unit() -> Result<()> {
        let temperatures = BTreeMap::from([
            ("choice".into(), Temperature::try_from(7.)?),
            ("choice:2".into(), Temperature::try_from(2.)?),
            ("choice:3-5".into(), Temperature::try_from(3.)?),
            ("choice:6-10".into(), Temperature::try_from(4.)?),
            ("choice:11+".into(), Temperature::try_from(5.)?),
        ]);
        for (count, expected) in [(2, 2.), (3, 3.), (5, 3.), (6, 4.), (10, 4.), (11, 5.)] {
            let readout = OmniReadout::prepare(&choice(count), OmniMedia::Text, &temperatures)?;
            assert_eq!(f64::from(readout.temperature), expected);
        }
        let type_only = BTreeMap::from([("choice".into(), Temperature::try_from(7.)?)]);
        assert_eq!(
            f64::from(OmniReadout::prepare(&choice(2), OmniMedia::Text, &type_only)?.temperature),
            7.
        );
        assert_eq!(
            f64::from(
                OmniReadout::prepare(&choice(2), OmniMedia::Text, &BTreeMap::new())?.temperature
            ),
            1.
        );
        let readout = OmniReadout::prepare(&choice(2), OmniMedia::Text, &temperatures)?;
        let p = readout.probabilities(&[0., 2., f32::NAN])?;
        assert!((p[0] - 0.2689414213699951).abs() < 1e-14);
        assert!((p[1] - 0.7310585786300049).abs() < 1e-14);
        Ok(())
    }

    #[test]
    fn media_bypasses_calibration_and_noul_reverses_after_normalization() -> Result<()> {
        let q = Question::Noul {
            instructions: "True?".into(),
            criteria: None,
        };
        let temperatures = BTreeMap::from([("noul:2".into(), Temperature::try_from(2.)?)]);
        let text =
            OmniReadout::prepare(&q, OmniMedia::Text, &temperatures)?.probabilities(&[0., 2.])?;
        assert!((text[0] - 0.7310585786300049).abs() < 1e-14);
        for media in [OmniMedia::Image, OmniMedia::Audio] {
            let p = OmniReadout::prepare(&q, media, &temperatures)?.probabilities(&[0., 2.])?;
            assert!((p[0] - 0.8807970779778823).abs() < 1e-14);
            assert_eq!(q.answer(&p)?.to_json()["noul"], p[0]);
        }
        Ok(())
    }

    #[test]
    fn invalid_real_marker_logits_fail_but_padding_is_not_an_option() -> Result<()> {
        assert!(OmniReadout::prepare(&choice(1), OmniMedia::Text, &BTreeMap::new()).is_err());
        let r = OmniReadout::prepare(&choice(2), OmniMedia::Text, &BTreeMap::new())?;
        for logits in [
            vec![],
            vec![1.],
            vec![f32::NAN, 0.],
            vec![0., f32::INFINITY],
        ] {
            assert!(r.probabilities(&logits).is_err());
        }
        assert_eq!(r.probabilities(&[0., 0., f32::NEG_INFINITY])?, [0.5, 0.5]);
        Ok(())
    }
}
