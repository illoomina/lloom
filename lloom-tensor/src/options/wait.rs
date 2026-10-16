//! Field-attached Wait grammar. Conversion keeps the stored vocabulary;
//! the field's existing parser remains the domain and capability check.

/// A duration's unit and checked scale into the stored microseconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DurationUnit {
    Microseconds,
    Milliseconds,
    Seconds,
}

impl DurationUnit {
    pub const ALL: [Self; 3] = [Self::Microseconds, Self::Milliseconds, Self::Seconds];

    pub const fn suffix(self) -> &'static str {
        match self {
            Self::Microseconds => "us",
            Self::Milliseconds => "ms",
            Self::Seconds => "s",
        }
    }

    pub const fn scale(self) -> u64 {
        match self {
            Self::Microseconds => 1,
            Self::Milliseconds => 1_000,
            Self::Seconds => 1_000_000,
        }
    }
}

/// Zero through u64::MAX microseconds; raw counts have no plus sign,
/// durations accept an optional plus and surrounding parameter whitespace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaitParameter {
    Microseconds,
}

impl WaitParameter {
    pub const fn maximum(self) -> u64 {
        match self {
            Self::Microseconds => u64::MAX,
        }
    }

    pub fn units(self) -> &'static [DurationUnit] {
        match self {
            Self::Microseconds => &DurationUnit::ALL,
        }
    }

    pub fn split_unit(self, s: &str) -> Option<(&str, DurationUnit)> {
        self.units()
            .iter()
            .find_map(|unit| s.strip_suffix(unit.suffix()).map(|n| (n, *unit)))
    }

    fn convert(self, arm: &str, parameter: Option<&str>) -> Result<String, String> {
        match self {
            Self::Microseconds => super::v1::with_us(arm, parameter),
        }
    }
}

/// A Wait site's literal arms and its optional microsecond parameter.
/// JSON null resets the field through the existing option layer; other
/// native JSON scalars are not part of this string grammar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WaitGrammar {
    pub literals: &'static [&'static str],
    pub parameter_arm: &'static str,
    pub parameter: WaitParameter,
}

impl WaitGrammar {
    /// Unknown stored spellings still reach the field's parser, which owns
    /// its named errors. Only the parameterized arm needs a conversion.
    pub fn to_stored(self, value: &str) -> Result<String, String> {
        let lower = value.trim().to_ascii_lowercase();
        let (arm, parameter) = match lower.split_once(':') {
            Some((arm, parameter)) => (arm, Some(parameter)),
            None => (lower.as_str(), None),
        };
        if arm == self.parameter_arm {
            self.parameter.convert(arm, parameter)
        } else {
            Ok(lower)
        }
    }
}

pub const READBACK_WAIT: WaitGrammar = WaitGrammar {
    literals: &[
        "spin",
        "spin-hint",
        #[cfg(target_arch = "aarch64")]
        "wfe",
    ],
    parameter_arm: "sleep",
    parameter: WaitParameter::Microseconds,
};
