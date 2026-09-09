//! The exclusion set (`design/docs/optimizer-design.md` §10a P2): which
//! transforms the offline numerical validation admitted, embedded at build
//! and consulted at construction — never a runtime obligation.
//!
//! `design/tools/fusion-validate/` runs every transform arm the harnesses
//! expose against the reference arm on one forced token sequence (#170) and
//! records, per transform, the four numbers the rule reads: the max f16 ulp
//! at the logits, the last/first growth ratio, the `overrides` count, and
//! whether the captured completion is identical to the reference's. The rule
//! is stated once (`verdict.py`'s docstring) and read twice: the tool derives
//! the verdict it writes, and [`ExclusionSet::parse`] re-derives it from the
//! numbers and refuses a file whose recorded verdict disagrees — so an edit by
//! hand to a verdict column is a load error, not a policy.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::OnceLock;

use lloom_tensor::{Error, Result};

/// The committed set, produced by the sweep recorded in
/// `design/measurements/issue-732-optimizer-p2.md`.
const EMBEDDED: &str = include_str!("exclusions.txt");

/// The admissibility rule's constants. One statement, in `verdict.py`; these
/// are its second reader.
pub const ULP_ADMIT: f64 = 32.0;
pub const RATIO_MAX: f64 = 2.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Admit,
    Exclude,
}

impl fmt::Display for Verdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Verdict::Admit => "admit",
            Verdict::Exclude => "exclude",
        })
    }
}

/// One validated transform.
#[derive(Debug, Clone, PartialEq)]
pub struct Validated {
    pub name: String,
    /// The class the transform was expected in (`bit-exact`, `rounding`,
    /// `reorder`, `tolerance`), as recorded — informational, the rule does
    /// not read it.
    pub class: String,
    pub max_f16_ulp: f64,
    pub ratio_last_first: f64,
    pub overrides: u64,
    pub text_identical: bool,
    pub verdict: Verdict,
}

/// The rule. Admissible iff the magnitude is within `ULP_ADMIT` f16 ulp, the
/// error does not grow (`ratio <= RATIO_MAX`), the arm would have sampled no
/// token differently, and its completion is the reference's.
pub fn rule(
    max_f16_ulp: f64,
    ratio_last_first: f64,
    overrides: u64,
    text_identical: bool,
) -> Verdict {
    if max_f16_ulp <= ULP_ADMIT && ratio_last_first <= RATIO_MAX && overrides == 0 && text_identical
    {
        Verdict::Admit
    } else {
        Verdict::Exclude
    }
}

/// The set: every validated transform by name.
#[derive(Debug, Clone, Default)]
pub struct ExclusionSet {
    entries: BTreeMap<String, Validated>,
}

impl ExclusionSet {
    /// Parse the file format `verdict.py` writes: `#` comments; one transform
    /// per line as `name class max_f16_ulp ratio overrides text_identical
    /// verdict`. A recorded verdict that the rule does not derive from the
    /// line's own numbers is refused.
    pub fn parse(text: &str) -> Result<ExclusionSet> {
        let mut entries = BTreeMap::new();
        for (lineno, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let f: Vec<&str> = line.split_whitespace().collect();
            if f.len() != 7 {
                return Err(Error::Msg(format!(
                    "exclusions.txt:{}: expected 7 fields, got {}",
                    lineno + 1,
                    f.len()
                )));
            }
            let num = |s: &str, what: &str| -> Result<f64> {
                s.parse::<f64>().map_err(|_| {
                    Error::Msg(format!(
                        "exclusions.txt:{}: {what} `{s}` is not a number",
                        lineno + 1
                    ))
                })
            };
            let max_f16_ulp = num(f[2], "max_f16_ulp")?;
            let ratio_last_first = num(f[3], "ratio_last_first")?;
            let overrides: u64 = f[4].parse().map_err(|_| {
                Error::Msg(format!(
                    "exclusions.txt:{}: overrides `{}` is not a count",
                    lineno + 1,
                    f[4]
                ))
            })?;
            let text_identical = match f[5] {
                "yes" => true,
                "no" => false,
                other => {
                    return Err(Error::Msg(format!(
                        "exclusions.txt:{}: text_identical must be yes|no, got `{other}`",
                        lineno + 1
                    )))
                }
            };
            let recorded = match f[6] {
                "admit" => Verdict::Admit,
                "exclude" => Verdict::Exclude,
                other => {
                    return Err(Error::Msg(format!(
                        "exclusions.txt:{}: verdict must be admit|exclude, got `{other}`",
                        lineno + 1
                    )))
                }
            };
            let derived = rule(max_f16_ulp, ratio_last_first, overrides, text_identical);
            if derived != recorded {
                return Err(Error::Msg(format!(
                    "exclusions.txt:{}: `{}` records `{recorded}` but the rule derives `{derived}` \
                     from ulp {max_f16_ulp} ratio {ratio_last_first} overrides {overrides} text \
                     {}; the file was edited or the rule moved",
                    lineno + 1,
                    f[0],
                    if text_identical { "identical" } else { "differs" }
                )));
            }
            let name = f[0].to_string();
            if entries.contains_key(&name) {
                return Err(Error::Msg(format!(
                    "exclusions.txt:{}: `{name}` listed twice",
                    lineno + 1
                )));
            }
            entries.insert(
                name.clone(),
                Validated {
                    name,
                    class: f[1].to_string(),
                    max_f16_ulp,
                    ratio_last_first,
                    overrides,
                    text_identical,
                    verdict: recorded,
                },
            );
        }
        Ok(ExclusionSet { entries })
    }

    pub fn get(&self, name: &str) -> Option<&Validated> {
        self.entries.get(name)
    }

    pub fn verdict(&self, name: &str) -> Option<Verdict> {
        self.entries.get(name).map(|v| v.verdict)
    }

    pub fn entries(&self) -> impl Iterator<Item = &Validated> {
        self.entries.values()
    }

    /// The names the set excludes.
    pub fn excluded(&self) -> Vec<String> {
        self.entries
            .values()
            .filter(|v| v.verdict == Verdict::Exclude)
            .map(|v| v.name.clone())
            .collect()
    }

    /// The transform name a catalogue fragment is validated under: the
    /// fused kernel's family, `gemv_<fragment>`.
    pub fn fragment_transform(fragment: &str) -> String {
        format!("gemv_{fragment}")
    }

    /// The transform name a `SiluImpl` arm is validated under.
    pub fn silu_transform(arm: &str) -> String {
        format!("silu_impl={arm}")
    }
}

/// The embedded set, parsed once. A malformed or hand-edited file is a panic
/// at first use — at model construction, never on the token path — because a
/// policy read from a file that contradicts its own rule is not a policy.
pub fn embedded() -> &'static ExclusionSet {
    static SET: OnceLock<ExclusionSet> = OnceLock::new();
    SET.get_or_init(|| match ExclusionSet::parse(EMBEDDED) {
        Ok(set) => set,
        Err(e) => panic!("optimizer: the embedded exclusion set is invalid: {e}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rule at the corpus's own measured points (§12.2i, §12.2l, #170).
    #[test]
    fn the_rule_at_the_measured_points() {
        // reorderings and roundings
        assert_eq!(rule(2.50, 0.889, 0, true), Verdict::Admit); // gemv_silu
        assert_eq!(rule(2.25, 1.524, 0, true), Verdict::Admit); // gemv_residual
        assert_eq!(rule(2.75, 1.000, 0, true), Verdict::Admit); // RotatingRing
        assert_eq!(rule(3.50, 0.941, 0, true), Verdict::Admit); // MathMode
        assert_eq!(rule(0.00, 1.000, 0, true), Verdict::Admit); // bit-exact
                                                                // the approximation ladder
        assert_eq!(rule(4.88, 0.692, 0, true), Verdict::Admit); // literal
        assert_eq!(rule(27.50, 0.886, 0, true), Verdict::Admit); // coarse4
        assert_eq!(rule(259.38, 0.573, 0, true), Verdict::Exclude); // coarse1
        assert_eq!(rule(500.55, 0.974, 0, true), Verdict::Exclude); // hardswish
        assert_eq!(rule(2307.00, 1.044, 0, false), Verdict::Exclude); // relu
                                                                      // a defect: large from the first position, sampled differently
        assert_eq!(rule(1840.06, 0.787, 139, false), Verdict::Exclude); // #128
                                                                        // each conjunct alone refuses
        assert_eq!(rule(1.0, 3.0, 0, true), Verdict::Exclude);
        assert_eq!(rule(1.0, 1.0, 1, true), Verdict::Exclude);
        assert_eq!(rule(1.0, 1.0, 0, false), Verdict::Exclude);
    }

    #[test]
    fn a_verdict_the_rule_does_not_derive_is_refused() {
        let ok = "gemv_silu rounding 2.50 0.889 0 yes admit\nsilu_impl=relu tolerance 2307.00 1.044 5 no exclude\n";
        let set = ExclusionSet::parse(ok).unwrap();
        assert_eq!(set.verdict("gemv_silu"), Some(Verdict::Admit));
        assert_eq!(set.excluded(), vec!["silu_impl=relu".to_string()]);
        let edited = "gemv_silu rounding 2.50 0.889 0 yes exclude\n";
        let err = ExclusionSet::parse(edited).unwrap_err().to_string();
        assert!(
            err.contains("records `exclude` but the rule derives `admit`"),
            "{err}"
        );
        let edited = "silu_impl=relu tolerance 2307.00 1.044 5 no admit\n";
        assert!(ExclusionSet::parse(edited).is_err());
        assert!(ExclusionSet::parse("a b c\n").is_err());
        assert!(
            ExclusionSet::parse("x rounding 1 1 0 yes admit\nx rounding 1 1 0 yes admit\n")
                .is_err()
        );
    }

    /// The embedded file parses, and it is not empty: an empty set would
    /// exclude nothing and pass every gate below vacuously.
    #[test]
    fn the_embedded_set_parses_and_is_not_empty() {
        let set = embedded();
        assert!(set.entries().count() >= 10, "{:?}", set.entries().count());
    }

    /// P2's gate, half one: the set excludes NONE of the fusions this
    /// repository ships — both epilogues, the conv fold, the rotating ring,
    /// the strided/fused views, the sdpa arm.
    #[test]
    fn the_set_excludes_no_shipped_fusion() {
        let set = embedded();
        for name in [
            "gemv_silu",
            "gemv_residual",
            "gemv_silu+gemv_residual",
            "conv_fused",
            "conv_state=rotating",
            "conv_state=ring",
            "qkv_layout=strided",
            "qkv_layout=fused",
            "attn=sdpa",
        ] {
            assert_eq!(
                set.verdict(name),
                Some(Verdict::Admit),
                "{name}: {:?}",
                set.get(name)
            );
        }
        // And the catalogue's fragments resolve to validated, admitted names.
        for f in crate::optimizer::catalogue() {
            let t = ExclusionSet::fragment_transform(f.name);
            assert_eq!(set.verdict(&t), Some(Verdict::Admit), "{t}");
        }
    }

    /// P2's gate, half two: the set contains every transform the corpus knows
    /// to be unsafe — §12.2l's degenerate `relu` and its gate-invisible
    /// degraded `hardswish`.
    #[test]
    fn the_set_contains_every_known_unsafe_transform() {
        let set = embedded();
        for name in ["silu_impl=relu", "silu_impl=hardswish"] {
            assert_eq!(
                set.verdict(name),
                Some(Verdict::Exclude),
                "{name}: {:?}",
                set.get(name)
            );
        }
        // The control read zero.
        let recip = set
            .get("silu_impl=recip")
            .expect("the recip control is in the set");
        assert_eq!(recip.max_f16_ulp, 0.0);
    }
}
