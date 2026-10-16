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

/// An additional parameterized spelling, including a stored-vocabulary alias.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WaitArm {
    pub spelling: &'static str,
    pub stored: &'static str,
    pub parameter: WaitParameter,
}

/// A Wait site's literal arms and its optional parameters.
/// JSON null resets the field through the existing option layer; other
/// native JSON scalars are not part of this string grammar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WaitGrammar {
    pub literals: &'static [&'static str],
    pub parameter_arm: &'static str,
    pub parameter: WaitParameter,
    pub additional_parameters: &'static [WaitArm],
}

impl WaitGrammar {
    /// Conversion and projection enumerate the same arms and domains.
    pub fn parameter_arms(self) -> impl Iterator<Item = WaitArm> {
        std::iter::once(WaitArm {
            spelling: self.parameter_arm,
            stored: self.parameter_arm,
            parameter: self.parameter,
        })
        .chain(self.additional_parameters.iter().copied())
    }

    /// Unknown stored spellings still reach the field's parser, which owns
    /// its named errors. Only parameterized arms need a conversion.
    pub fn to_stored(self, value: &str) -> Result<String, String> {
        let lower = value.trim().to_ascii_lowercase();
        let (arm, parameter) = match lower.split_once(':') {
            Some((arm, parameter)) => (arm, Some(parameter)),
            None => (lower.as_str(), None),
        };
        if let Some(found) = self.parameter_arms().find(|a| a.spelling == arm) {
            found.parameter.convert(found.stored, parameter)
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
    additional_parameters: &[],
};

pub const HOST_FEED_WAIT: WaitGrammar = WaitGrammar {
    literals: READBACK_WAIT.literals,
    parameter_arm: "sleep",
    parameter: WaitParameter::Microseconds,
    additional_parameters: &[
        WaitArm {
            spelling: "adaptive",
            stored: "on-demand",
            parameter: WaitParameter::Microseconds,
        },
        WaitArm {
            spelling: "on-demand",
            stored: "on-demand",
            parameter: WaitParameter::Microseconds,
        },
        WaitArm {
            spelling: "event",
            stored: "event",
            parameter: WaitParameter::Microseconds,
        },
    ],
};
