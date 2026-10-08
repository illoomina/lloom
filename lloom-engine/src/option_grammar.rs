//! The value grammar, over the whole tree.
//!
//! `lloom_tensor::options` holds one parser per value type (a bool, an enum's
//! arm, a count, a number, a duration); every field parses through them, so
//! a spelling means one thing on every flag, dotted key, JSON value and env
//! name. The test below holds that over the option index rather than over the
//! source: every key of a type accepts and refuses the same spellings through
//! every way a value reaches it.
//!
//! [`legacy_notes`] is the other half: an env variable whose value selected
//! something else before the variables joined the tree says so once, at
//! startup.

use crate::EngineOptions;
use lloom_tensor::options::{before, BeforeRow};

/// The front-end's legacy env names whose values meant something else
/// before they joined the option tree.
pub const LEGACY_BEFORE: &[BeforeRow] = &[("LFM2_MODEL_DIR", "load.model", before::path_verbatim)];

/// One line per legacy env variable `env` sets to a value whose meaning
/// differs from what it selected before the variables joined the option
/// tree, over both library crates' groups and the front-end's own.
pub fn legacy_notes(env: &dyn Fn(&str) -> Option<String>) -> Vec<String> {
    let mut out = lloom_tensor::tuning::legacy_notes(env);
    out.extend(lloom::tuning::legacy_notes(env));
    out.extend(lloom_tensor::options::legacy_notes(
        LEGACY_BEFORE,
        EngineOptions::META,
        env,
        &|alone| {
            let mut o = EngineOptions::default();
            o.from_env_with(alone)?;
            serde_json::to_value(&o).map_err(|e| e.to_string())
        },
    ));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::option_tree::{index, Key, Leaf};
    use lloom_tensor::options::{canonical_env, KnobMeta, FALSE_SPELLINGS, TRUE_SPELLINGS};
    use serde_json::Value;

    /// A key's value type, read off its declaration.
    #[derive(Debug, Clone, PartialEq)]
    enum Ty {
        Bool {
            optional: bool,
        },
        Enum {
            arms: Vec<&'static str>,
            optional: bool,
        },
        Count {
            optional: bool,
        },
        Number {
            optional: bool,
        },
        /// A path, a name, a list: taken verbatim.
        Text,
    }

    fn ty_of(m: &KnobMeta) -> Ty {
        let optional = m.ty.starts_with("Option<");
        let inner =
            m.ty.strip_prefix("Option<")
                .and_then(|t| t.strip_suffix('>'))
                .unwrap_or(m.ty);
        if m.is_bool || inner == "bool" {
            return Ty::Bool { optional };
        }
        let simple = |a: &&'static str| {
            !a.is_empty()
                && a.chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        };
        let numeric = |a: &&'static str| a.chars().all(|c| c.is_ascii_digit());
        let arms: Vec<&'static str> = if m.variants.is_empty() {
            m.forms.to_vec()
        } else {
            m.variants.to_vec()
        };
        match inner {
            "usize" | "u64" | "u32" | "u16" | "u8" => Ty::Count { optional },
            "f64" => Ty::Number { optional },
            _ if !arms.is_empty() && arms.iter().any(|a| simple(a) && !numeric(a)) => Ty::Enum {
                arms: arms
                    .into_iter()
                    .filter(|a| simple(a) && *a != "none")
                    .collect(),
                optional,
            },
            _ => Ty::Text,
        }
    }

    /// One way a value reaches a leaf without a command line.
    #[derive(Debug, Clone)]
    enum Via {
        /// `--set` / a dotted key: the key itself, which converts and inverts.
        Key(String),
        /// A legacy dotted spelling (the storage path, the flat axis keys).
        Dotted(String),
        /// The key's `LLOOM_<KEY>`, which inverts as the key does.
        Canonical(String),
        /// A legacy env name.
        Env(&'static str),
        /// A JSON string at the key, as `--config` and `--<group>-config` read it.
        Json(String),
    }

    impl Via {
        fn inverts(&self) -> bool {
            matches!(self, Via::Key(_) | Via::Canonical(_) | Via::Json(_))
        }

        fn write(&self, value: &str) -> Result<EngineOptions, String> {
            self.write_json(Value::String(value.into()))
        }

        fn write_json(&self, value: Value) -> Result<EngineOptions, String> {
            let idx = index();
            let mut o = EngineOptions::default();
            let text = || match &value {
                Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            match self {
                Via::Key(k) | Via::Dotted(k) => idx.set(&mut o, k, &text())?,
                Via::Canonical(var) => {
                    let v = text();
                    o.from_env_with(&|n| (n == var).then(|| v.clone()))?
                }
                Via::Env(var) => {
                    let v = text();
                    o.from_env_with(&|n| (n == *var).then(|| v.clone()))?
                }
                Via::Json(k) => {
                    let doc = k.rsplit('.').fold(
                        value.clone(),
                        |inner, seg| serde_json::json!({ seg: inner }),
                    );
                    idx.merge(&mut o, &doc)?
                }
            }
            Ok(o)
        }
    }

    /// Every way a value reaches `leaf` of `key` without a command line.
    fn vias(key: &Key, leaf: &Leaf) -> Vec<Via> {
        let idx = index();
        let mut out = Vec::new();
        let targeted = key.targets.iter().any(|t| t.storage == leaf.storage);
        if targeted {
            out.push(Via::Key(key.path.clone()));
            out.push(Via::Json(key.path.clone()));
            out.push(Via::Canonical(canonical_env(&key.path)));
        }
        for s in idx.spellings() {
            if s != key.path && idx.storage_of(s) == Some(leaf.storage.as_str()) {
                out.push(Via::Dotted(s.to_string()));
            }
        }
        if let Some(e) = leaf.meta.env {
            out.push(Via::Env(e));
        }
        out
    }

    /// A leaf's stored value as a bool, whether it is stored as one or as an
    /// axis spelling.
    fn as_bool(v: &Value) -> Option<bool> {
        match v {
            Value::Bool(b) => Some(*b),
            Value::String(s) => lloom_tensor::options::parse_bool(s).ok(),
            _ => None,
        }
    }

    fn leaf_value(o: &EngineOptions, storage: &str) -> Value {
        let v = serde_json::to_value(o).expect("the groups serialize");
        storage
            .split('.')
            .try_fold(&v, |v, seg| v.get(seg))
            .cloned()
            .unwrap_or(Value::Null)
    }

    const NOT_BOOLS: &[&str] = &[
        "", " ", "2", "-1", "tru", "yess", "onn", "o", "enable", "true1", "nope",
    ];

    /// Every bool key accepts the same spellings and refuses the same
    /// spellings, through every way a value reaches it, and an optional one
    /// is unset by `none`. A key that parses its bool any other way fails
    /// here with its name and the spelling.
    #[test]
    fn every_bool_key_takes_one_grammar() {
        let idx = index();
        let mut bad: Vec<String> = Vec::new();
        let mut checked = 0usize;
        for key in idx.keys.values().filter(|k| k.compound.is_none()) {
            for leaf in key.targets.iter().chain(&key.legacy) {
                let Ty::Bool { optional } = ty_of(leaf.meta) else {
                    continue;
                };
                // An axis is optional in the tree (unset = the preset's arm).
                let optional = optional || leaf.default.is_null();
                for via in vias(key, leaf) {
                    let flip = via.inverts() && leaf.meta.invert;
                    let mut accepts = |spelling: &str, want: bool| {
                        checked += 1;
                        match via.write(spelling) {
                            Ok(o) => {
                                let got = as_bool(&leaf_value(&o, &leaf.storage));
                                if got != Some(want ^ flip) {
                                    bad.push(format!(
                                        "{} via {via:?}: `{spelling}` stored {got:?}, want {}",
                                        key.path,
                                        want ^ flip
                                    ));
                                }
                            }
                            Err(e) => bad.push(format!(
                                "{} via {via:?}: `{spelling}` refused: {e}",
                                key.path
                            )),
                        }
                    };
                    for (spellings, want) in [(TRUE_SPELLINGS, true), (FALSE_SPELLINGS, false)] {
                        for s in spellings {
                            accepts(s, want);
                            accepts(&s.to_ascii_uppercase(), want);
                            accepts(&format!(" {s} "), want);
                        }
                    }
                    if matches!(via, Via::Json(_)) {
                        for want in [true, false] {
                            checked += 1;
                            match via.write_json(Value::Bool(want)) {
                                Ok(o) => {
                                    let got = as_bool(&leaf_value(&o, &leaf.storage));
                                    if got != Some(want ^ flip) {
                                        bad.push(format!(
                                            "{} via {via:?}: JSON {want} stored {got:?}",
                                            key.path
                                        ));
                                    }
                                }
                                Err(e) => bad.push(format!(
                                    "{} via {via:?}: JSON {want} refused: {e}",
                                    key.path
                                )),
                            }
                        }
                    }
                    for s in NOT_BOOLS {
                        checked += 1;
                        if via.write(s).is_ok() {
                            bad.push(format!("{} via {via:?}: `{s}` accepted", key.path));
                        }
                    }
                    let none = via.write("NONE");
                    checked += 1;
                    match (optional, none) {
                        (true, Ok(o)) if leaf_value(&o, &leaf.storage).is_null() => {}
                        (false, Err(_)) => {}
                        (true, other) => bad.push(format!(
                            "{} via {via:?}: `NONE` on an optional bool: {:?}",
                            key.path,
                            other.map(|o| leaf_value(&o, &leaf.storage))
                        )),
                        (false, Ok(_)) => {
                            bad.push(format!("{} via {via:?}: `NONE` accepted", key.path))
                        }
                    }
                }
            }
        }
        eprintln!("[{checked} bool spellings checked]");
        assert!(checked > 10_000, "only {checked} spellings checked");
        assert!(
            bad.is_empty(),
            "{} failures:\n{}",
            bad.len(),
            bad.join("\n")
        );
    }

    /// Every enum, count and number key: an arm in any case and with spaces
    /// around it is the arm; a count or number with spaces around it is the
    /// count; the empty value and a value of no arm are refused; an optional
    /// one is unset by `none`.
    #[test]
    fn every_typed_key_takes_one_grammar() {
        let idx = index();
        let mut bad: Vec<String> = Vec::new();
        let mut checked = 0usize;
        for key in idx.keys.values().filter(|k| k.compound.is_none()) {
            for leaf in key.targets.iter().chain(&key.legacy) {
                let ty = ty_of(leaf.meta);
                let optional = match &ty {
                    Ty::Enum { optional, .. }
                    | Ty::Count { optional }
                    | Ty::Number { optional } => *optional || leaf.default.is_null(),
                    _ => continue,
                };
                let samples: Vec<String> = match &ty {
                    Ty::Enum { arms, .. } => arms.iter().map(|a| a.to_string()).collect(),
                    Ty::Count { .. } => {
                        let mut v: Vec<String> = Vec::new();
                        if leaf.default.is_u64() {
                            v.push(leaf.default.to_string());
                        }
                        v.extend(
                            ["1", "2", "4", "8", "16", "32", "64", "256"]
                                .iter()
                                .map(|s| s.to_string()),
                        );
                        v
                    }
                    Ty::Number { .. } => {
                        let mut v: Vec<String> = Vec::new();
                        if leaf.default.is_number() {
                            v.push(leaf.default.to_string());
                        }
                        v.extend(["1", "0.5", "1.5", "2"].iter().map(|s| s.to_string()));
                        v
                    }
                    _ => unreachable!(),
                };
                for via in vias(key, leaf) {
                    let mut took_one = false;
                    for sample in &samples {
                        let Ok(plain) = via.write(sample) else {
                            if matches!(ty, Ty::Enum { .. }) {
                                bad.push(format!(
                                    "{} via {via:?}: its arm `{sample}` is refused",
                                    key.path
                                ));
                            }
                            continue;
                        };
                        took_one = true;
                        let want = leaf_value(&plain, &leaf.storage);
                        let mut forms = vec![format!(" {sample} ")];
                        if matches!(ty, Ty::Enum { .. }) {
                            forms.push(sample.to_ascii_uppercase());
                        }
                        for form in forms {
                            checked += 1;
                            match via.write(&form) {
                                Ok(o) if leaf_value(&o, &leaf.storage) == want => {}
                                Ok(o) => bad.push(format!(
                                    "{} via {via:?}: `{form}` stored {}, `{sample}` {want}",
                                    key.path,
                                    leaf_value(&o, &leaf.storage)
                                )),
                                Err(e) => bad.push(format!(
                                    "{} via {via:?}: `{form}` refused: {e}",
                                    key.path
                                )),
                            }
                        }
                        if !matches!(ty, Ty::Enum { .. }) {
                            break;
                        }
                    }
                    if !took_one {
                        bad.push(format!("{} via {via:?}: no sample is accepted", key.path));
                    }
                    let refused: &[&str] = match ty {
                        Ty::Enum { .. } => &["", " ", "no-such-arm"],
                        Ty::Count { .. } => &["", " ", "x", "-1", "1.5", "4x"],
                        _ => &["", " ", "x", "nan", "inf", "1..5"],
                    };
                    for s in refused {
                        checked += 1;
                        if let Ok(o) = via.write(s) {
                            bad.push(format!(
                                "{} via {via:?}: `{s}` accepted (stored {})",
                                key.path,
                                leaf_value(&o, &leaf.storage)
                            ));
                        }
                    }
                    if optional {
                        checked += 1;
                        match via.write("NONE") {
                            Ok(o) => {
                                let got = leaf_value(&o, &leaf.storage);
                                // `none` unsets, or is the arm a key declares
                                // by that name (`weights.mmap_advice`).
                                let arm = leaf.meta.variants.contains(&"none") && got == "none";
                                if got != leaf.default && !got.is_null() && !arm {
                                    bad.push(format!(
                                        "{} via {via:?}: `NONE` stored {got}",
                                        key.path
                                    ));
                                }
                            }
                            Err(e) => {
                                bad.push(format!("{} via {via:?}: `NONE` refused: {e}", key.path))
                            }
                        }
                    }
                }
            }
        }
        eprintln!("[{checked} typed spellings checked]");
        assert!(checked > 1_000, "only {checked} spellings checked");
        assert!(
            bad.is_empty(),
            "{} failures:\n{}",
            bad.len(),
            bad.join("\n")
        );
    }

    #[test]
    fn a_legacy_value_whose_meaning_moved_says_so_and_one_that_did_not_is_silent() {
        let notes = |pairs: &[(&str, &str)]| {
            let pairs: Vec<(String, String)> = pairs
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect();
            legacy_notes(&move |n: &str| pairs.iter().find(|(k, _)| k == n).map(|(_, v)| v.clone()))
        };
        let moved = notes(&[
            ("LLOOM_WG_SYNC", "0"),
            ("QWEN4EXP_MASK", "true"),
            ("CANDLE_METAL_ENABLE_FAST_MATH", "on"),
            ("LLOOM_148_SHARE_TRACE", "0"),
            ("CANDLE_MMAP_ADVICE", ""),
        ]);
        assert_eq!(moved.len(), 5, "{moved:#?}");
        let line = |var: &str| {
            moved
                .iter()
                .find(|l| l.contains(&format!("{var}=")))
                .unwrap_or_else(|| panic!("no line for {var}: {moved:#?}"))
        };
        assert_eq!(
            line("LLOOM_WG_SYNC"),
            "[options] LLOOM_WG_SYNC=\"0\" selected weights.groups.sync = true before the option tree; it now selects false"
        );
        assert!(
            line("QWEN4EXP_MASK").ends_with("= false before the option tree; it now selects true")
        );
        assert!(line("CANDLE_METAL_ENABLE_FAST_MATH")
            .contains("backend.metal.compiler.fast_math = false"));
        assert!(line("LLOOM_148_SHARE_TRACE").ends_with("it now selects false"));
        assert!(line("CANDLE_MMAP_ADVICE").contains("it now selects nothing: it is refused"));
        let same = notes(&[
            ("LLOOM_WG_SYNC", "1"),
            ("QWEN4EXP_MASK", "1"),
            ("QWEN4EXP_REPLAY", "device"),
            ("CANDLE_METAL_ENABLE_FAST_MATH", "0"),
            ("CANDLE_METAL_EAGER_RESIDENCY_RELEASE", "1"),
            ("LLOOM_148_POOL_PAGES", "16"),
            ("LLOOM_383_TILE", "32x32"),
            ("CANDLE_MMAP_ADVICE", "none"),
            ("LLOOM_MOE_PAIR", "on"),
        ]);
        assert!(same.is_empty(), "{same:#?}");
    }

    /// Every row of the three tables names a legacy env name of the field it
    /// says, at that field's storage path in the tree.
    #[test]
    fn the_tables_name_real_fields() {
        let idx = index();
        let tables: [(&str, &[BeforeRow]); 3] = [
            ("tuning.", lloom_tensor::tuning::LEGACY_BEFORE),
            ("tuning.", lloom::tuning::LEGACY_BEFORE),
            ("", LEGACY_BEFORE),
        ];
        let mut n = 0;
        for (prefix, table) in tables {
            for (var, storage, _) in table {
                let storage = format!("{prefix}{storage}");
                let key = idx
                    .key_of_storage(&storage)
                    .unwrap_or_else(|| panic!("{storage} is not a field"));
                let leaf = key
                    .targets
                    .iter()
                    .chain(&key.legacy)
                    .find(|l| l.storage == storage)
                    .expect("found by storage");
                assert_eq!(leaf.meta.env, Some(*var), "{storage}");
                n += 1;
            }
        }
        assert!(n >= 25, "{n} rows");
    }
}
