//! The exclusion set (`design/docs/optimizer-design.md` §10a P2, §10c): which
//! transforms the offline numerical validation admitted, embedded at build
//! and consulted at construction — never a runtime obligation.
//!
//! `design/tools/fusion-validate/` runs every transform arm the harnesses
//! expose against the reference arm on one forced token sequence (#170) and
//! records, per transform, the four numbers the rule reads — the max f16 ulp
//! at the logits, the last/first growth ratio, the `overrides` count, and
//! whether the captured completion is identical to the reference's — and two
//! declarations the rule does not read but every consumer must: the
//! [`TargetFpModel`] the transform's kernel computes under, and the
//! [`RegionKind`] the transform rewrites. The rule is stated once
//! (`verdict.py`'s docstring) and read twice: the tool derives the verdict it
//! writes, and [`ExclusionSet::parse`] re-derives it from the numbers and
//! refuses a file whose recorded verdict disagrees — so an edit by hand to a
//! verdict column is a load error, not a policy.
//!
//! A verdict is looked up UNDER a model ([`ExclusionSet::lookup`]): one
//! recorded for `simd_sum`'s tree does not transfer to the serial
//! `conv_fold`, and a consumer asking under a different model is told the
//! transform is unvalidated there rather than handed the other kernel's
//! answer (§5a). The region is what makes composition visible: two admitted
//! transforms on one straight-line region compose by addition; two on one
//! recurrence do not, and want their own joint entry.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::OnceLock;

use lloom_tensor::{Error, Result};

use super::numeric::{RegionKind, TargetFpModel};

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
    /// The fp model the transform's kernel computed under when the numbers
    /// were taken. The verdict is a fact about this model only.
    pub model: TargetFpModel,
    /// The region the transform rewrites: a verdict on a recurrence was
    /// measured through it (500 steps, the growth ratio) and does not compose
    /// with another transform on the same recurrence by addition.
    pub region: RegionKind,
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

/// What the set says about a transform, asked under a model.
#[derive(Debug, Clone, PartialEq)]
pub enum Lookup<'a> {
    /// Validated under the model asked for; the entry.
    Validated(&'a Validated),
    /// Validated, but under another model: the entry is not evidence here.
    ModelMismatch(&'a Validated),
    /// The set has no entry for the name.
    Unvalidated,
}

/// The set: every validated transform by name.
#[derive(Debug, Clone, Default)]
pub struct ExclusionSet {
    entries: BTreeMap<String, Validated>,
}

impl ExclusionSet {
    /// Parse the file format `verdict.py` writes: `#` comments; one transform
    /// per line as `name class max_f16_ulp ratio overrides text_identical
    /// verdict fp_model region`. A recorded verdict that the rule does not
    /// derive from the line's own numbers is refused, and so is a line
    /// without its model or region — an entry recorded without the key it
    /// was derived under has to be re-derived to learn it (§5a).
    pub fn parse(text: &str) -> Result<ExclusionSet> {
        let mut entries = BTreeMap::new();
        for (lineno, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let f: Vec<&str> = line.split_whitespace().collect();
            if f.len() != 9 {
                return Err(Error::Msg(format!(
                    "exclusions.txt:{}: expected 9 fields (name class ulp ratio overrides \
                     text verdict fp_model region), got {}",
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
            let model = TargetFpModel::parse(f[7])
                .map_err(|e| Error::Msg(format!("exclusions.txt:{}: fp model: {e}", lineno + 1)))?;
            let region = RegionKind::parse(f[8])
                .map_err(|e| Error::Msg(format!("exclusions.txt:{}: region: {e}", lineno + 1)))?;
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
                    model,
                    region,
                },
            );
        }
        Ok(ExclusionSet { entries })
    }

    pub fn get(&self, name: &str) -> Option<&Validated> {
        self.entries.get(name)
    }

    /// The recorded verdict by name, whatever model it was recorded under.
    /// For a decision, ask [`Self::lookup`] under the model the consumer
    /// computes in; this is the raw record.
    pub fn verdict(&self, name: &str) -> Option<Verdict> {
        self.entries.get(name).map(|v| v.verdict)
    }

    /// The verdict for `name` UNDER `model`: the entry if it was recorded
    /// under that model, a mismatch if under another, nothing if absent.
    pub fn lookup(&self, name: &str, model: &TargetFpModel) -> Lookup<'_> {
        match self.entries.get(name) {
            None => Lookup::Unvalidated,
            Some(v) if v.model == *model => Lookup::Validated(v),
            Some(v) => Lookup::ModelMismatch(v),
        }
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

    /// The name a JOINT validation of several transforms is recorded under:
    /// the names joined with `+`, in the order the sweep spells them. Two
    /// transforms on one recurrence want this entry; their separate verdicts
    /// do not compose there (§5a).
    pub fn joint_transform(names: &[&str]) -> String {
        names.join("+")
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
    use crate::optimizer::numeric::{Accum, ReductionTree};

    const GEMV: &str = "fma=contract,acc=f32,tree=simd,denorm32=flush,denorm16=preserve";
    const UNARY: &str = "fma=contract,acc=none,tree=none,denorm32=flush,denorm16=preserve";

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
        let ok = format!(
            "gemv_silu rounding 2.50 0.889 0 yes admit {GEMV} straight\n\
             silu_impl=relu tolerance 2307.00 1.044 5 no exclude {UNARY} straight\n"
        );
        let set = ExclusionSet::parse(&ok).unwrap();
        assert_eq!(set.verdict("gemv_silu"), Some(Verdict::Admit));
        assert_eq!(set.excluded(), vec!["silu_impl=relu".to_string()]);
        let edited = format!("gemv_silu rounding 2.50 0.889 0 yes exclude {GEMV} straight\n");
        let err = ExclusionSet::parse(&edited).unwrap_err().to_string();
        assert!(
            err.contains("records `exclude` but the rule derives `admit`"),
            "{err}"
        );
        let edited =
            format!("silu_impl=relu tolerance 2307.00 1.044 5 no admit {UNARY} straight\n");
        assert!(ExclusionSet::parse(&edited).is_err());
        assert!(ExclusionSet::parse("a b c\n").is_err());
        assert!(ExclusionSet::parse(&format!(
            "x rounding 1 1 0 yes admit {GEMV} straight\nx rounding 1 1 0 yes admit {GEMV} straight\n"
        ))
        .is_err());
    }

    /// An entry without its model or region is refused: recorded without
    /// the key, it would have to be re-derived to learn which model it
    /// belonged to (§5a's retrofit cost, paid never rather than later).
    #[test]
    fn an_entry_without_its_model_or_region_is_refused() {
        let seven = "gemv_silu rounding 2.50 0.889 0 yes admit\n";
        let err = ExclusionSet::parse(seven).unwrap_err().to_string();
        assert!(err.contains("expected 9 fields"), "{err}");
        let no_region = format!("gemv_silu rounding 2.50 0.889 0 yes admit {GEMV}\n");
        assert!(ExclusionSet::parse(&no_region).is_err());
        let bad_model = "gemv_silu rounding 2.50 0.889 0 yes admit fma=contract,acc=f32 straight\n";
        let err = ExclusionSet::parse(bad_model).unwrap_err().to_string();
        assert!(err.contains("fp model"), "{err}");
        let bad_region = format!("gemv_silu rounding 2.50 0.889 0 yes admit {GEMV} ring\n");
        let err = ExclusionSet::parse(&bad_region).unwrap_err().to_string();
        assert!(err.contains("region"), "{err}");
    }

    /// A verdict recorded under one model is not evidence under another:
    /// asked for the GEMV's tree, the serial fold's entry is a mismatch, not
    /// an answer.
    #[test]
    fn a_verdict_recorded_under_another_model_does_not_transfer() {
        let text = "conv_fused reorder 2.50 1.231 0 yes admit fma=contract,acc=f32,tree=serial,denorm32=flush,denorm16=preserve conv-ring\n";
        let set = ExclusionSet::parse(text).unwrap();
        let serial = TargetFpModel::parse(
            "fma=contract,acc=f32,tree=serial,denorm32=flush,denorm16=preserve",
        )
        .unwrap();
        let rotated = TargetFpModel::parse(
            "fma=contract,acc=f32,tree=serial-rot,denorm32=flush,denorm16=preserve",
        )
        .unwrap();
        assert!(
            matches!(set.lookup("conv_fused", &serial), Lookup::Validated(v) if v.verdict == Verdict::Admit)
        );
        match set.lookup("conv_fused", &rotated) {
            Lookup::ModelMismatch(v) => {
                assert_eq!(v.model.tree, ReductionTree::Serial);
                assert_eq!(v.region, RegionKind::ConvRing);
            }
            other => panic!("expected a model mismatch, got {other:?}"),
        }
        assert_eq!(set.lookup("nothing", &serial), Lookup::Unvalidated);
        assert_eq!(ExclusionSet::joint_transform(&["a", "b"]), "a+b");
    }

    /// The embedded file parses, and it is not empty: an empty set would
    /// exclude nothing and pass every gate below vacuously.
    #[test]
    fn the_embedded_set_parses_and_is_not_empty() {
        let set = embedded();
        assert!(set.entries().count() >= 10, "{:?}", set.entries().count());
    }

    /// Every embedded entry's recorded model is the model this crate
    /// declares for the kernel that transform runs — the tool's table and
    /// [`TargetFpModel::of_family`] are two readers of one declaration and
    /// may not drift; and every stateful transform says so.
    #[test]
    fn every_entry_carries_the_model_its_kernel_declares_and_its_region() {
        let set = embedded();
        let build = crate::optimizer::numeric::build_model();
        for v in set.entries() {
            let family = transform_family(&v.name)
                .unwrap_or_else(|| panic!("`{}`: no kernel family declared in this test", v.name));
            let declared = TargetFpModel::of_family(family)
                .unwrap_or_else(|| panic!("`{}`: `{family}` not in of_family", v.name));
            assert_eq!(
                v.model, declared,
                "`{}` recorded under {} but `{family}` computes under {declared}",
                v.name, v.model
            );
            assert_eq!(v.model.fma, build.fma);
            assert_eq!(
                (v.model.denormals_f32, v.model.denormals_f16),
                (build.denormals_f32, build.denormals_f16),
                "`{}`: recorded under another build",
                v.name
            );
            let stateful = v.name.starts_with("conv_")
                || v.name.starts_with("attn=")
                || v.name.starts_with("kv_quant");
            assert_eq!(
                v.region.is_stateful(),
                stateful,
                "`{}` region {}",
                v.name,
                v.region
            );
        }
        // Absolute references for the two the design names (§5a).
        assert_eq!(
            set.get("gemv_silu").unwrap().model.tree,
            ReductionTree::Simd
        );
        assert_eq!(
            set.get("conv_fused").unwrap().model.tree,
            ReductionTree::Serial
        );
        assert_eq!(set.get("conv_fused").unwrap().model.accum, Accum::F32);
        assert_eq!(
            set.get("conv_state=rotating").unwrap().model.accum,
            Accum::Store
        );
        assert_eq!(set.get("conv_fused").unwrap().region, RegionKind::ConvRing);
        assert_eq!(set.get("attn=sdpa").unwrap().region, RegionKind::Kv);
    }

    /// The kernel family each recorded transform runs (the sweep's arms);
    /// the test-side half of the declaration `verdict.py` writes.
    fn transform_family(name: &str) -> Option<&'static str> {
        Some(match name {
            "gemv_silu" | "gemv_residual" | "gemv_silu+gemv_residual" => "gemv",
            "conv_fused" => "shortconv_step",
            "conv_fused+conv_state=rotating" => "shortconv_ring",
            "conv_state=rotating" | "conv_state=ring" => "fast_sum",
            "qkv_layout=strided" => "rmsnorm",
            "qkv_layout=fused" => "rmsnorm_rope",
            "attn=sdpa" => "sdpa_vector",
            "kv_quant=int8" => "dequantize_kv",
            n if n.starts_with("silu_impl=") => "unary",
            _ => return None,
        })
    }

    /// P2's gate, half one: the set excludes NONE of the fusions this
    /// repository ships — both epilogues, the conv fold, the rotating ring,
    /// the strided/fused views, the sdpa arm, and the crown's fused ring.
    #[test]
    fn the_set_excludes_no_shipped_fusion() {
        let set = embedded();
        for name in [
            "gemv_silu",
            "gemv_residual",
            "gemv_silu+gemv_residual",
            "conv_fused",
            "conv_fused+conv_state=rotating",
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
        // And the catalogue's fragments resolve to validated, admitted names
        // under the model they compose into.
        for f in crate::optimizer::catalogue() {
            let t = ExclusionSet::fragment_transform(f.name);
            assert!(
                matches!(set.lookup(&t, &f.model), Lookup::Validated(v) if v.verdict == Verdict::Admit),
                "{t} under {}: {:?}",
                f.model,
                set.get(&t)
            );
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
