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

/// The conversion domain, distinct from the field parser's raw-count domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaitParameter {
    Microseconds,
    Milliseconds,
    RawCount { trim: bool },
}

impl WaitParameter {
    pub const fn maximum(self) -> u64 {
        match self {
            Self::Microseconds | Self::Milliseconds | Self::RawCount { .. } => u64::MAX,
        }
    }

    pub fn units(self) -> &'static [DurationUnit] {
        match self {
            Self::Microseconds | Self::Milliseconds => &DurationUnit::ALL,
            Self::RawCount { .. } => &[],
        }
    }

    pub const fn raw_plus(self) -> bool {
        matches!(self, Self::RawCount { .. })
    }

    pub const fn raw_trim(self) -> bool {
        matches!(self, Self::RawCount { trim: true })
    }

    /// Explicit durations are checked in microseconds before whole-ms division.
    pub const fn duration_quantum(self, unit: DurationUnit) -> u64 {
        match (self, unit) {
            (Self::Milliseconds, DurationUnit::Microseconds) => 1_000,
            _ => 1,
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
            Self::Milliseconds => super::v1::with_ms(arm, parameter),
            Self::RawCount { .. } => Ok(match parameter {
                Some(p) => format!("{arm}:{p}"),
                None => arm.to_string(),
            }),
        }
    }
}

/// Native counts are checked against integer bounds, never a rounded float
/// bound. The floating branch converts the represented integral value only;
/// precision discarded by JSON deserialization cannot be recovered here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CountDomain {
    PositiveUsize,
}

impl CountDomain {
    pub const fn minimum(self) -> u64 {
        1
    }

    pub const fn maximum(self) -> u64 {
        usize::MAX as u64
    }

    pub fn normalize(self, number: &serde_json::Number) -> Result<String, String> {
        let integer = number.as_u64().or_else(|| {
            let f = number.as_f64()?;
            if !f.is_finite() || f < 1.0 || f.fract() != 0.0 {
                return None;
            }
            let precision_probe =
                serde_json::from_str::<serde_json::Number>("1.00000000000000001").ok();
            if precision_probe != serde_json::Number::from_f64(1.0) {
                return None;
            }
            if serde_json::Number::from_f64(f).as_ref() != Some(number) {
                return None;
            }
            format!("{f:.0}").parse::<u64>().ok()
        });
        integer
            .filter(|n| *n >= self.minimum() && *n <= self.maximum())
            .and_then(|n| usize::try_from(n).ok())
            .map(|n| n.to_string())
            .ok_or_else(|| {
                super::parse_count::<usize>(&number.to_string())
                    .err()
                    .unwrap_or_else(|| "must be >= 1".into())
            })
    }
}

/// A wait stored as a bare count, also reachable as a native JSON number.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WaitCount {
    pub domain: CountDomain,
    pub default: u64,
}

/// An additional parameterized spelling, including a stored-vocabulary alias.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WaitArm {
    pub spelling: &'static str,
    pub stored: &'static str,
    pub parameter: WaitParameter,
}

/// A Wait site's literal arms and its optional parameters.
/// JSON null resets through the existing option layer. Only count-backed
/// sites also declare a native JSON number domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WaitGrammar {
    pub literals: &'static [&'static str],
    pub count: Option<WaitCount>,
    pub literal_aliases: &'static [(&'static str, &'static str)],
    pub parser: Option<&'static str>,
    pub parameter_arm: &'static str,
    pub parameter_stored: &'static str,
    pub parameter: WaitParameter,
    pub additional_parameters: &'static [WaitArm],
}

impl WaitGrammar {
    /// Conversion and projection enumerate the same arms and domains.
    pub fn parameter_arms(self) -> impl Iterator<Item = WaitArm> {
        std::iter::once(WaitArm {
            spelling: self.parameter_arm,
            stored: self.parameter_stored,
            parameter: self.parameter,
        })
        .chain(self.additional_parameters.iter().copied())
    }

    /// Unknown stored spellings still reach the field's parser, which owns
    /// its named errors. Only parameterized arms need a conversion.
    pub fn to_stored(self, value: &str) -> Result<String, String> {
        let lower = value.trim().to_ascii_lowercase();
        if let Some((_, stored)) = self
            .literal_aliases
            .iter()
            .find(|(s, _)| *s == lower.as_str())
        {
            return Ok((*stored).to_string());
        }
        let (arm, parameter) = match lower.split_once(':') {
            Some((arm, parameter)) => (arm, Some(parameter)),
            None => (lower.as_str(), None),
        };
        if let Some(found) = self.parameter_arms().find(|a| a.spelling == arm) {
            let stored = found.parameter.convert(found.stored, parameter)?;
            match self.count {
                Some(count) if found.stored.is_empty() => Ok(match parameter {
                    None => count.default.to_string(),
                    Some(_) => stored
                        .strip_prefix(':')
                        .expect("empty stored arm")
                        .to_string(),
                }),
                _ => Ok(stored),
            }
        } else {
            Ok(lower)
        }
    }
}

pub const READBACK_WAIT: WaitGrammar = WaitGrammar {
    count: None,
    literal_aliases: &[],
    parser: None,
    literals: &[
        "spin",
        "spin-hint",
        #[cfg(target_arch = "aarch64")]
        "wfe",
    ],
    parameter_arm: "sleep",
    parameter_stored: "sleep",
    parameter: WaitParameter::Microseconds,
    additional_parameters: &[],
};

pub const HOST_FEED_WAIT: WaitGrammar = WaitGrammar {
    count: None,
    literal_aliases: &[],
    parser: None,
    literals: READBACK_WAIT.literals,
    parameter_arm: "sleep",
    parameter_stored: "sleep",
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

/// Raw legacy suffixes are parsed directly with Metal and trimmed by the
/// declaration parser without Metal. Conversion leaves those bytes intact.
pub const COMPLETION_WAIT: WaitGrammar = WaitGrammar {
    count: None,
    literals: &["synchronize", "shared-event"],
    literal_aliases: &[("block", "synchronize"), ("event", "shared-event")],
    #[cfg(feature = "metal")]
    parser: Some("CompletionWait::from_str_arm (metal)"),
    #[cfg(not(feature = "metal"))]
    parser: Some("declared_arm (no metal)"),
    parameter_arm: "event",
    parameter_stored: "shared-event-wait",
    parameter: WaitParameter::Milliseconds,
    additional_parameters: &[
        WaitArm {
            spelling: "event-sleep",
            stored: "shared-event-sleep",
            parameter: WaitParameter::Microseconds,
        },
        WaitArm {
            spelling: "event-listener",
            stored: "shared-event-listener",
            parameter: WaitParameter::Milliseconds,
        },
        WaitArm {
            spelling: "shared-event-sleep",
            stored: "shared-event-sleep",
            parameter: WaitParameter::RawCount {
                trim: !cfg!(feature = "metal"),
            },
        },
        WaitArm {
            spelling: "shared-event-wait",
            stored: "shared-event-wait",
            parameter: WaitParameter::RawCount {
                trim: !cfg!(feature = "metal"),
            },
        },
        WaitArm {
            spelling: "shared-event-listener",
            stored: "shared-event-listener",
            parameter: WaitParameter::RawCount {
                trim: !cfg!(feature = "metal"),
            },
        },
    ],
};

/// The field parser keeps the positive-usize text contract; native numbers
/// carry that same domain through the typed JSON ingress and projection.
pub const FETCH_WAIT: WaitGrammar = WaitGrammar {
    count: Some(WaitCount {
        domain: CountDomain::PositiveUsize,
        default: 50,
    }),
    literals: &[],
    literal_aliases: &[],
    parser: None,
    parameter_arm: "sleep",
    parameter_stored: "",
    parameter: WaitParameter::Microseconds,
    additional_parameters: &[],
};
