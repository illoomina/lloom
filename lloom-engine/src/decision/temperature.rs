use lloom_tensor::{bail, Result};
use serde::{Deserialize, Serialize};

/// Option-logit temperature, applied before option normalization, not to the
/// resulting probabilities. Checkpoint readers and API adapters share validation.
#[derive(Clone, Copy, Debug, PartialEq, Deserialize, Serialize)]
#[serde(try_from = "f64", into = "f64")]
pub struct Temperature(f64);

impl TryFrom<f64> for Temperature {
    type Error = lloom_tensor::Error;
    fn try_from(value: f64) -> Result<Self> {
        if !value.is_finite() || value <= 0.0 {
            bail!("decision temperature must be finite and positive");
        }
        Ok(Self(value))
    }
}

impl From<Temperature> for f64 {
    fn from(value: Temperature) -> Self {
        value.0
    }
}

#[cfg(test)]
mod tests {
    use super::Temperature;

    #[test]
    fn json_temperature_requires_finite_positive_number() {
        for input in ["null", "true", "\"2\"", "[]", "{}", "0", "-1", "1e999"] {
            assert!(
                serde_json::from_str::<Temperature>(input).is_err(),
                "{input}"
            );
        }
        assert_eq!(
            f64::from(serde_json::from_str::<Temperature>("2.5").unwrap()),
            2.5
        );
    }
}
