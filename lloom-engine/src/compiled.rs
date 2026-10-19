//! What this binary was compiled with, for the options that need it: one
//! table of compiled capabilities, read from the crate that holds each
//! `cfg`, and the setup check over the option tree's declarations
//! (`#[opt(needs = "…")]`, `KnobMeta::needs`).
//!
//! A key whose whole effect needs a capability the build lacks is refused
//! when it is set, naming the build that has it; a key that loses only part
//! of its output is noted. Never a flag accepted into an empty trace. A cold
//! path: the options are checked once, at setup.

use crate::option_tree::index;
use crate::EngineOptions;
use lloom_tensor::options::KnobMeta;

/// One compiled capability of this binary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Capability {
    /// The name a key's `needs` uses.
    pub name: &'static str,
    /// Whether the code it names is in this binary.
    pub compiled: bool,
    /// The `lloom-engine` feature that compiles it in.
    pub feature: &'static str,
    /// Whether that feature is on in this build (a debug build has
    /// `profile`'s code without the feature).
    pub feature_on: bool,
    pub what: &'static str,
}

/// This binary's capabilities.
pub fn this_build() -> Vec<Capability> {
    vec![
        Capability {
            name: "profile",
            compiled: lloom::PROFILE_INSTRUMENTS,
            feature: "profile",
            feature_on: cfg!(feature = "profile"),
            what: "the decode loop's per-step instrumentation (debug builds, or an optimized build with the feature)",
        },
        Capability {
            name: "metal",
            compiled: cfg!(feature = "metal"),
            feature: "metal",
            feature_on: cfg!(feature = "metal"),
            what: "the Metal backend",
        },
        Capability {
            name: "cuda",
            compiled: cfg!(feature = "cuda"),
            feature: "cuda",
            feature_on: cfg!(feature = "cuda"),
            what: "the CUDA backend",
        },
    ]
}

/// The cargo line that builds this binary with `needed` added.
pub fn build_line(caps: &[Capability], needed: &str) -> String {
    let mut features = vec!["serve"];
    features.extend(caps.iter().filter(|c| c.feature_on).map(|c| c.feature));
    if let Some(c) = caps.iter().find(|c| c.name == needed) {
        if !features.contains(&c.feature) {
            features.push(c.feature);
        }
    }
    format!(
        "cargo build --release -p lloom-engine --bin lloom --features {}",
        features.join(",")
    )
}

/// The options against the build: `Err` names every key set whose whole
/// effect needs a capability this build lacks, with the build that has it;
/// `Ok` carries a note per key set that loses part of its output.
pub fn check(o: &EngineOptions, caps: &[Capability]) -> Result<Vec<String>, String> {
    let idx = index();
    let now = idx.flat(o);
    let defaults = idx.flat(&EngineOptions::default());
    verdict(
        idx.keys
            .values()
            .filter(|k| k.compound.is_none() && now.get(&k.path) != defaults.get(&k.path))
            .map(|k| (k.path.as_str(), k.meta())),
        caps,
    )
}

/// [`check`]'s verdict over the declarations of the keys that are set.
pub fn verdict<'a>(
    set: impl IntoIterator<Item = (&'a str, &'static KnobMeta)>,
    caps: &[Capability],
) -> Result<Vec<String>, String> {
    let (mut refused, mut notes) = (Vec::new(), Vec::new());
    for (path, m) in set {
        if m.needs.is_empty() {
            continue;
        }
        let Some(cap) = caps.iter().find(|c| c.name == m.needs) else {
            refused.push(format!(
                "{path} needs `{}`, a capability this binary does not know",
                m.needs
            ));
            continue;
        };
        if cap.compiled {
            continue;
        }
        let line = build_line(caps, cap.name);
        if m.needs_part.is_empty() {
            refused.push(format!(
                "{path} needs the `{}` build, which this binary is not: {line}",
                cap.name
            ));
        } else {
            notes.push(format!(
                "[options] {path} is set, but {} needs the `{}` build, which this binary is not ({line}); it prints the rest",
                m.needs_part, cap.name
            ));
        }
    }
    if refused.is_empty() {
        Ok(notes)
    } else {
        Err(refused.join("; "))
    }
}

/// `lloom options --compiled`: one line per capability.
pub fn table_lines(caps: &[Capability]) -> Vec<String> {
    caps.iter()
        .map(|c| {
            format!(
                "{name}  {state}  feature={feature}  {what}",
                name = c.name,
                state = if c.compiled { "compiled" } else { "absent" },
                feature = c.feature,
                what = c.what
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plain(caps: &[Capability]) -> Vec<Capability> {
        caps.iter()
            .map(|c| Capability {
                compiled: c.name != "profile" && c.compiled,
                feature_on: c.name != "profile" && c.feature_on,
                ..*c
            })
            .collect()
    }

    /// Every key's `needs` names a capability of the table, and the profile
    /// capability is the gate the decode loop's sites compile under.
    #[test]
    fn every_declared_need_is_a_capability() {
        let caps = this_build();
        let mut n = 0;
        for key in index().keys.values().filter(|k| k.compound.is_none()) {
            let m = key.meta();
            if m.needs.is_empty() {
                continue;
            }
            n += 1;
            assert!(
                caps.iter().any(|c| c.name == m.needs),
                "{} needs `{}`, which the table does not hold",
                key.path,
                m.needs
            );
        }
        assert!(n >= 3, "{n} keys declare a need");
        let profile = caps.iter().find(|c| c.name == "profile").unwrap();
        assert_eq!(
            profile.compiled,
            cfg!(any(debug_assertions, feature = "profile")),
            "the engine's view of `profile` is the decode loop's gate"
        );
    }

    /// On a build without `profile`: a key that loses part of its output is
    /// noted, naming the part and the build line; a key at its default is
    /// silent; the line names this build's features plus `profile`.
    #[test]
    fn a_partial_need_is_noted_on_a_plain_build() {
        let caps = plain(&this_build());
        let mut o = EngineOptions::default();
        assert_eq!(check(&o, &caps), Ok(vec![]), "nothing set, nothing said");
        index().set(&mut o, "diagnostics.verbose", "true").unwrap();
        let notes = check(&o, &caps).expect("a partial need is not refused");
        assert_eq!(notes.len(), 1, "{notes:?}");
        let n = &notes[0];
        assert!(
            n.contains("diagnostics.verbose")
                && n.contains("the step-time percentiles")
                && n.contains("`profile` build")
                && n.contains("cargo build --release -p lloom-engine --bin lloom --features serve")
                && n.ends_with("it prints the rest"),
            "{n}"
        );
        assert!(n.contains(",profile)"), "the line adds the feature: {n}");
        let all = this_build();
        assert_eq!(
            check(&o, &all).unwrap().len(),
            usize::from(!lloom::PROFILE_INSTRUMENTS),
            "a build with the capability says nothing"
        );
    }

    /// A key whose whole effect needs the capability is refused by name on
    /// a build without it, with the build line; set on a build with it, or
    /// left at its default, it passes. No shipping key is whole-`profile`
    /// today, so the refusal arm runs over a group of the test's own,
    /// through the same declaration the derive emits.
    #[test]
    fn a_whole_need_is_refused_naming_the_build() {
        #[derive(
            lloom_macros::Options,
            serde::Serialize,
            serde::Deserialize,
            Default,
            Debug,
            Clone,
            PartialEq,
        )]
        #[serde(default)]
        struct Probe {
            /// A trace whose recorder only the profile build has.
            #[opt(
                bool,
                path = "diagnostics.trace.probe",
                status = "instrument",
                needs = "profile"
            )]
            trace: bool,
        }
        let m: &'static KnobMeta = &Probe::META[0];
        assert_eq!((m.needs, m.needs_part), ("profile", ""));
        let caps = plain(&this_build());
        let e =
            verdict([("diagnostics.trace.probe", m)], &caps).expect_err("refused on a plain build");
        assert!(
            e.starts_with("diagnostics.trace.probe needs the `profile` build, which this binary is not: cargo build --release -p lloom-engine --bin lloom --features serve")
                && e.ends_with(",profile"),
            "{e}"
        );
        assert_eq!(
            verdict([], &caps),
            Ok(vec![]),
            "a key left off is not in the set"
        );
        let with: Vec<Capability> = caps
            .iter()
            .map(|c| Capability {
                compiled: true,
                ..*c
            })
            .collect();
        assert_eq!(verdict([("diagnostics.trace.probe", m)], &with), Ok(vec![]));
        let unknown = KnobMeta {
            needs: "telepathy",
            ..*m
        };
        let unknown: &'static KnobMeta = Box::leak(Box::new(unknown));
        assert!(verdict([("x", unknown)], &with)
            .unwrap_err()
            .contains("does not know"));
    }
}
