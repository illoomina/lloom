//! The option tree: every option of [`EngineOptions`] under its key, and the
//! index from every spelling a user has ever typed — the storage path, the
//! legacy dotted keys — to the leaf it names.
//!
//! The groups keep their storage shape (the library crates read
//! `tuning().qmv.q6k_nr0`); the tree is the interface. It is built once from
//! the derive's `META`, so the keys, the flags, the env names, the schema and
//! `lloom options` come from the field declarations and cannot drift from
//! them. Nothing here runs per token.
//!
//! A write through a KEY targets every field declared under it (a merged
//! option fans out: `kernels.fusion.norm_in_gemv` sets LFM2's and
//! Qwen4-exp's), converts the value into the stored vocabulary, and resets
//! the fields that are its legacy aliases. A write through a STORAGE path
//! writes that one field, as it always did.

use crate::options::EngineOptions;
use lloom_tensor::options::{KnobMeta, Status};
use serde_json::{Map, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

/// One field under a key: where it is stored, its declaration, its default.
#[derive(Debug, Clone)]
pub struct Leaf {
    pub storage: String,
    pub meta: &'static KnobMeta,
    pub default: Value,
}

/// One key of the tree.
#[derive(Debug, Clone)]
pub struct Key {
    pub path: String,
    /// The fields a write to the key sets.
    pub targets: Vec<Leaf>,
    /// The fields whose spellings are this key's aliases; a write to the key
    /// returns them to their defaults, and a non-default one is what the key
    /// renders (it decides at its read site).
    pub legacy: Vec<Leaf>,
    /// A group's shorthand spelling (`decode.readback = ring-hostdrain`): the
    /// storage path of the group whose parser takes it. Write-only.
    pub compound: Option<String>,
    /// The compound's own declaration (its group field's).
    pub own: Option<&'static KnobMeta>,
}

impl Key {
    /// The declaration the key's type, status and description come from.
    pub fn meta(&self) -> &'static KnobMeta {
        self.targets
            .first()
            .or(self.legacy.first())
            .map(|l| l.meta)
            .or(self.own)
            .expect("a key has a field")
    }

    pub fn status(&self) -> Status {
        self.meta().status
    }

    /// The dedicated flag, if the key has one.
    pub fn cli(&self) -> Option<&'static str> {
        self.targets
            .iter()
            .map(|l| l.meta.cli)
            .find(|c| !c.is_empty())
    }

    /// Every flag spelling that reaches the key, the dedicated one first.
    pub fn flags(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        let mut push = |f: &str| {
            let f = format!("--{f}");
            if !out.contains(&f) {
                out.push(f);
            }
        };
        if let Some(c) = self.cli() {
            push(c);
        }
        for l in self.targets.iter().chain(&self.legacy) {
            if !l.meta.flag.is_empty() {
                push(l.meta.flag);
            }
            for a in l.meta.aliases {
                push(a);
            }
            for (sw, _) in l.meta.switches {
                push(sw);
            }
        }
        out
    }

    /// The key's canonical env name.
    pub fn env(&self) -> String {
        lloom_tensor::options::canonical_env(&self.path)
    }

    /// The legacy env names that reach the key.
    pub fn legacy_env(&self) -> Vec<&'static str> {
        let mut out: Vec<&'static str> = Vec::new();
        for l in self.targets.iter().chain(&self.legacy) {
            if let Some(e) = l.meta.env {
                if !out.contains(&e) {
                    out.push(e);
                }
            }
        }
        out
    }

    /// The legacy dotted spellings of the key.
    pub fn legacy_keys(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for l in self.targets.iter().chain(&self.legacy) {
            if l.meta.v0 && l.storage != self.path && !out.contains(&l.storage) {
                out.push(l.storage.clone());
            }
            for k in l.meta.legacy_keys {
                if !out.iter().any(|o| o == k) {
                    out.push(k.to_string());
                }
            }
        }
        out
    }

    /// A merged key over a field that is unset by default (an axis) is
    /// unset until one of its fields moves.
    pub fn unset_by_default(&self) -> bool {
        self.targets.len() > 1 && self.targets.iter().any(|t| t.default.is_null())
    }

    fn map_like(&self) -> bool {
        self.meta().ty.contains("Map")
    }
}

/// What a dotted spelling names.
#[derive(Debug, Clone)]
enum Spelling {
    /// A key: the write fans out and converts.
    Key(String),
    /// A legacy spelling of one field: the write is the field's own.
    Storage { key: String, storage: String },
}

/// The index over the whole tree.
#[derive(Debug)]
pub struct Index {
    pub keys: BTreeMap<String, Key>,
    spellings: BTreeMap<String, Spelling>,
}

/// The process's index, built once.
pub fn index() -> &'static Index {
    static INDEX: OnceLock<Index> = OnceLock::new();
    INDEX.get_or_init(|| Index::build().unwrap_or_else(|e| panic!("the option tree: {e}")))
}

fn get<'a>(v: &'a Value, path: &str) -> Option<&'a Value> {
    path.split('.').try_fold(v, |v, seg| v.get(seg))
}

/// A JSON object holding `value` at the dotted `path`.
fn fragment(path: &str, value: Value) -> Value {
    path.rsplit('.').fold(value, |inner, seg| {
        let mut m = Map::new();
        m.insert(seg.to_string(), inner);
        Value::Object(m)
    })
}

fn insert(tree: &mut Value, path: &str, value: Value) {
    let mut node = tree;
    let segs: Vec<&str> = path.split('.').collect();
    for seg in &segs[..segs.len() - 1] {
        node = node
            .as_object_mut()
            .expect("a group")
            .entry(seg.to_string())
            .or_insert_with(|| Value::Object(Map::new()));
    }
    node.as_object_mut()
        .expect("a group")
        .insert(segs[segs.len() - 1].to_string(), value);
}

/// A scalar in its flag spelling.
fn scalar(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// A stored value in its key's vocabulary.
pub fn to_key(meta: &KnobMeta, v: &Value) -> Value {
    if v.is_null() {
        return Value::Null;
    }
    if meta.invert {
        if let Some(b) = v.as_bool() {
            return Value::Bool(!b);
        }
    }
    let s = scalar(v);
    if let Some((k, _)) = meta.values.iter().find(|(_, st)| *st == s) {
        return Value::String(k.to_string());
    }
    if let Some(f) = meta.to_v1 {
        return Value::String(f(&s));
    }
    v.clone()
}

impl Index {
    /// The index over [`EngineOptions::META`]; `Err` names a spelling that
    /// would name two fields.
    pub fn build() -> Result<Index, String> {
        let defaults = serde_json::to_value(EngineOptions::default()).map_err(|e| e.to_string())?;
        let mut idx = Index {
            keys: BTreeMap::new(),
            spellings: BTreeMap::new(),
        };
        idx.walk("", EngineOptions::META, &defaults)?;
        // The model group's axes were also spelled flat (`model.kv-append`,
        // what `ModelConfig::set` takes).
        let axes: Vec<(String, String, String)> = idx
            .keys
            .values()
            .flat_map(|k| {
                k.targets
                    .iter()
                    .map(move |l| (k.path.clone(), l.storage.clone()))
            })
            .filter_map(|(key, storage)| {
                let flat = storage.strip_prefix("model.axes.")?;
                Some((format!("model.{flat}"), key, storage))
            })
            .collect();
        for (spelling, key, storage) in axes {
            idx.spellings
                .insert(spelling, Spelling::Storage { key, storage });
        }
        // Every key is its own spelling; a legacy spelling may not shadow
        // another key.
        for path in idx.keys.keys() {
            if let Some(Spelling::Storage { key, storage }) = idx.spellings.get(path) {
                if key != path {
                    return Err(format!(
                        "`{path}` is a key and the legacy spelling of `{key}` (field {storage})"
                    ));
                }
            }
        }
        let keys: Vec<String> = idx.keys.keys().cloned().collect();
        for k in keys {
            idx.spellings.insert(k.clone(), Spelling::Key(k));
        }
        Ok(idx)
    }

    fn walk(
        &mut self,
        prefix: &str,
        metas: &'static [KnobMeta],
        defaults: &Value,
    ) -> Result<(), String> {
        for m in metas {
            let storage = if prefix.is_empty() {
                m.key.to_string()
            } else {
                format!("{prefix}.{}", m.key)
            };
            if let Some(sub) = m.nested {
                if m.compound {
                    let path = if m.path.is_empty() {
                        storage.clone()
                    } else {
                        m.path.to_string()
                    };
                    self.keys.insert(
                        path.clone(),
                        Key {
                            path,
                            targets: vec![],
                            legacy: vec![],
                            compound: Some(storage.clone()),
                            own: Some(m),
                        },
                    );
                }
                self.walk(&storage, sub, defaults)?;
                continue;
            }
            let path = if m.path.is_empty() {
                storage.clone()
            } else {
                m.path.to_string()
            };
            let leaf = Leaf {
                storage: storage.clone(),
                meta: m,
                default: get(defaults, &storage).cloned().unwrap_or(Value::Null),
            };
            let key = self.keys.entry(path.clone()).or_insert_with(|| Key {
                path: path.clone(),
                targets: vec![],
                legacy: vec![],
                compound: None,
                own: None,
            });
            if key.compound.is_some() {
                return Err(format!(
                    "`{path}` is a group's shorthand and a field ({storage})"
                ));
            }
            if m.legacy_only {
                key.legacy.push(leaf);
            } else {
                key.targets.push(leaf);
            }
            let mut spell = |s: String| -> Result<(), String> {
                let new = Spelling::Storage {
                    key: path.clone(),
                    storage: storage.clone(),
                };
                if let Some(Spelling::Storage { storage: other, .. }) = self.spellings.get(&s) {
                    if *other != storage {
                        return Err(format!("`{s}` would name both {other} and {storage}"));
                    }
                }
                self.spellings.insert(s, new);
                Ok(())
            };
            if m.v0 && storage != path {
                spell(storage.clone())?;
            }
            for k in m.legacy_keys {
                spell(k.to_string())?;
            }
        }
        Ok(())
    }

    /// The key a spelling names, if it names one.
    pub fn key_of(&self, spelling: &str) -> Option<&Key> {
        match self.spellings.get(spelling)? {
            Spelling::Key(k) | Spelling::Storage { key: k, .. } => self.keys.get(k),
        }
    }

    /// The field a legacy dotted spelling writes; `None` for a key or an
    /// unknown spelling.
    pub fn storage_of(&self, spelling: &str) -> Option<&str> {
        match self.spellings.get(spelling)? {
            Spelling::Storage { storage, .. } => Some(storage),
            Spelling::Key(_) => None,
        }
    }

    /// The key a field (by storage path) belongs to.
    pub fn key_of_storage(&self, storage: &str) -> Option<&Key> {
        self.keys.values().find(|k| {
            k.targets
                .iter()
                .chain(&k.legacy)
                .any(|l| l.storage == storage)
        })
    }

    /// Every dotted spelling the index accepts.
    pub fn spellings(&self) -> impl Iterator<Item = &str> {
        self.spellings.keys().map(String::as_str)
    }

    /// Set one option by any dotted spelling, the value in its flag
    /// spelling or its key's.
    pub fn set(&self, o: &mut EngineOptions, spelling: &str, value: &str) -> Result<(), String> {
        self.write(
            o,
            spelling,
            &Value::String(value.to_string()),
            &mut BTreeSet::new(),
        )
    }

    fn write(
        &self,
        o: &mut EngineOptions,
        spelling: &str,
        value: &Value,
        written: &mut BTreeSet<String>,
    ) -> Result<(), String> {
        let Some(found) = self.spellings.get(spelling) else {
            return Err(format!("unknown key `{spelling}`"));
        };
        let mut once = |storage: &str| -> Result<(), String> {
            if !written.insert(storage.to_string()) {
                return Err(format!(
                    "`{spelling}` sets a field another spelling in the same layer set ({storage})"
                ));
            }
            Ok(())
        };
        match found {
            Spelling::Storage { storage, .. } => {
                once(storage)?;
                let leaf = self.leaf(storage).expect("indexed");
                write_leaf(o, leaf, value, false)
            }
            Spelling::Key(k) => {
                let key = &self.keys[k];
                if let Some(storage) = &key.compound {
                    let Value::String(s) = value else {
                        return Err(format!("`{k}` takes a shorthand spelling here"));
                    };
                    return o.set(storage, s);
                }
                // `none` is `null` for a merged key that is unset by default
                // (it renders `null` until a field moves): every field
                // returns to its own default.
                let null = Value::Null;
                let value = match value {
                    Value::String(s)
                        if key.unset_by_default() && lloom_tensor::options::is_none(s) =>
                    {
                        &null
                    }
                    v => v,
                };
                for t in &key.targets {
                    once(&t.storage)?;
                    write_leaf(o, t, value, true)?;
                }
                for l in &key.legacy {
                    o.merge_json(&fragment(&l.storage, l.default.clone()))?;
                }
                Ok(())
            }
        }
    }

    fn leaf(&self, storage: &str) -> Option<&Leaf> {
        self.keys
            .values()
            .flat_map(|k| k.targets.iter().chain(&k.legacy))
            .find(|l| l.storage == storage)
    }

    /// Merge a JSON document in the tree's shape (or any legacy shape a key
    /// spells): `$schema` is skipped, a leaf is written by its key or legacy
    /// spelling, a group recurses. Two spellings of one field in one document
    /// are refused.
    pub fn merge(&self, o: &mut EngineOptions, json: &Value) -> Result<(), String> {
        self.merge_written(o, json).map(|_| ())
    }

    /// [`Self::merge`], returning the storage paths the document wrote.
    pub fn merge_written(
        &self,
        o: &mut EngineOptions,
        json: &Value,
    ) -> Result<BTreeSet<String>, String> {
        let mut written = BTreeSet::new();
        self.merge_at(o, "", json, None, &mut written)?;
        Ok(written)
    }

    /// `--<group>-config '{json}'`: the object is the group's; a member the
    /// group does not name is tried under `fallback` (the group's storage
    /// name, for `--generation-config`'s `sampling` keys).
    pub fn merge_group(
        &self,
        o: &mut EngineOptions,
        group: &str,
        json: &Value,
        fallback: Option<&str>,
    ) -> Result<(), String> {
        self.merge_group_written(o, group, json, fallback)
            .map(|_| ())
    }

    /// [`Self::merge_group`], returning the storage paths the object wrote.
    pub fn merge_group_written(
        &self,
        o: &mut EngineOptions,
        group: &str,
        json: &Value,
        fallback: Option<&str>,
    ) -> Result<BTreeSet<String>, String> {
        let Some(map) = json.as_object() else {
            return Err("expected a JSON object".into());
        };
        let mut written = BTreeSet::new();
        for (k, v) in map {
            let wrapped = fragment(k, v.clone());
            match self.merge_at(o, group, &wrapped, None, &mut written) {
                Err(e) if e.starts_with("unknown key") && fallback.is_some() => {
                    self.merge_at(o, fallback.unwrap(), &wrapped, None, &mut written)?
                }
                other => other?,
            }
        }
        Ok(written)
    }

    fn merge_at(
        &self,
        o: &mut EngineOptions,
        prefix: &str,
        json: &Value,
        _parent: Option<&str>,
        written: &mut BTreeSet<String>,
    ) -> Result<(), String> {
        let Some(map) = json.as_object() else {
            return Err(format!("`{prefix}`: expected a JSON object"));
        };
        for (k, v) in map {
            if prefix.is_empty() && k == "$schema" {
                continue;
            }
            let path = if prefix.is_empty() {
                k.clone()
            } else {
                format!("{prefix}.{k}")
            };
            match self.key_of(&path) {
                // An object under a key's spelling is a group of the same
                // name (`decode.readback`, `executor.arena`) unless the key
                // takes a map.
                Some(key) if v.is_object() && !key.map_like() => {
                    if self.is_group(&path) {
                        self.merge_at(o, &path, v, Some(prefix), written)?
                    } else {
                        return Err(format!("`{path}`: expected a value, got an object"));
                    }
                }
                Some(_) => self.write(o, &path, v, written).map_err(|e| {
                    if e.starts_with('`') {
                        e
                    } else {
                        format!("{path}: {e}")
                    }
                })?,
                None if v.is_object() && self.is_group(&path) => {
                    self.merge_at(o, &path, v, Some(prefix), written)?
                }
                None => return Err(format!("unknown key `{path}`")),
            }
        }
        Ok(())
    }

    /// Whether some spelling lies under `path`.
    pub fn is_group(&self, path: &str) -> bool {
        let p = format!("{path}.");
        self.spellings
            .range(p.clone()..)
            .next()
            .is_some_and(|(s, _)| s.starts_with(&p))
    }

    /// Every key's value, in its key's vocabulary.
    pub fn flat(&self, o: &EngineOptions) -> BTreeMap<String, Value> {
        let stored = serde_json::to_value(o).expect("the groups serialize");
        self.keys
            .values()
            .filter(|k| k.compound.is_none())
            .map(|k| (k.path.clone(), self.value_of(k, &stored)))
            .collect()
    }

    /// The tree: one object per group, every key's value.
    pub fn render(&self, o: &EngineOptions) -> Value {
        let mut tree = Value::Object(Map::new());
        for (path, v) in self.flat(o) {
            insert(&mut tree, &path, v);
        }
        tree
    }

    /// A key's value from the stored tree: a non-default legacy alias first
    /// (it decides at its read site), else the targets — the one that left
    /// its default, or, for a merged list, every one's entries.
    pub fn value_of(&self, key: &Key, stored: &Value) -> Value {
        let at = |l: &Leaf| get(stored, &l.storage).cloned().unwrap_or(Value::Null);
        for l in &key.legacy {
            let v = at(l);
            if v != l.default {
                return to_key(l.meta, &v);
            }
        }
        let vals: Vec<(Value, bool)> = key
            .targets
            .iter()
            .map(|t| {
                let v = at(t);
                let moved = v != t.default;
                (to_key(t.meta, &v), moved)
            })
            .collect();
        let moved: Vec<&Value> = vals.iter().filter(|(_, m)| *m).map(|(v, _)| v).collect();
        match moved.as_slice() {
            [] if key.unset_by_default() => Value::Null,
            [] => vals.first().map(|(v, _)| v.clone()).unwrap_or(Value::Null),
            [one] => (*one).clone(),
            many if many.iter().all(|v| *v == many[0]) => many[0].clone(),
            many if many.iter().all(|v| v.is_string()) => {
                let mut tokens: Vec<String> = Vec::new();
                for v in many {
                    for t in v.as_str().unwrap_or("").split(',').map(str::trim) {
                        if !t.is_empty() && !tokens.iter().any(|x| x == t) {
                            tokens.push(t.to_string());
                        }
                    }
                }
                Value::String(tokens.join(","))
            }
            many => many[0].clone(),
        }
    }

    /// The keys a group (or `""`, every group) holds, in order.
    pub fn keys_under<'a>(&'a self, group: &'a str) -> impl Iterator<Item = &'a Key> + 'a {
        self.keys.values().filter(move |k| {
            group.is_empty() || k.path == group || k.path.starts_with(&format!("{group}."))
        })
    }
}

/// `null` unsets: `None` for an optional field, the default for any other
/// (the fields a merged key shares with an optional one).
fn write_leaf(
    o: &mut EngineOptions,
    leaf: &Leaf,
    value: &Value,
    as_key: bool,
) -> Result<(), String> {
    match value {
        Value::Null => o.merge_json(&fragment(&leaf.storage, leaf.default.clone())),
        Value::Array(_) | Value::Object(_) => o.merge_json(&fragment(&leaf.storage, value.clone())),
        v => {
            let mut s = scalar(v);
            if as_key && leaf.meta.invert {
                s = (!lloom_tensor::options::parse_bool(&s).map_err(|e| {
                    let key = if leaf.meta.path.is_empty() {
                        leaf.storage.as_str()
                    } else {
                        leaf.meta.path
                    };
                    format!("`{key}`: {e}")
                })?)
                .to_string();
            }
            o.set(&leaf.storage, &s)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverted_invalid_index_writes_name_the_key_and_leave_options_unchanged() {
        let idx = index();
        let mut checked = 0;
        for key in idx
            .keys
            .values()
            .filter(|key| key.targets.iter().any(|l| l.meta.invert))
        {
            for seed in ["true", "false"] {
                let mut o = EngineOptions::default();
                idx.set(&mut o, &key.path, seed).unwrap();
                let before = o.clone();
                for value in ["", " \t\r\n ", "maybe", "2"] {
                    let error = idx.set(&mut o, &key.path, value).unwrap_err();
                    assert_eq!(
                        error,
                        format!(
                            "`{}`: {}",
                            key.path,
                            lloom_tensor::options::parse_bool(value).unwrap_err()
                        )
                    );
                    assert_eq!(o, before, "{}: {value:?}", key.path);
                    checked += 1;
                }
            }
        }
        assert!(checked > 0, "no inverted keys checked");
        eprintln!("[{checked} inverted Index::set refusals preserve options]");
    }

    #[test]
    fn inverted_invalid_group_json_names_the_key_and_leaves_options_unchanged() {
        let idx = index();
        let mut checked = 0;
        for key in idx
            .keys
            .values()
            .filter(|key| key.targets.iter().any(|l| l.meta.invert))
        {
            let (group, rest) = key.path.split_once('.').unwrap();
            for seed in ["true", "false"] {
                let mut o = EngineOptions::default();
                idx.set(&mut o, &key.path, seed).unwrap();
                let before = o.clone();
                for value in ["", " \t\r\n ", "maybe", "2"] {
                    let error = idx
                        .merge_group(
                            &mut o,
                            group,
                            &fragment(rest, Value::String(value.into())),
                            None,
                        )
                        .unwrap_err();
                    assert_eq!(
                        error,
                        format!(
                            "`{}`: {}",
                            key.path,
                            lloom_tensor::options::parse_bool(value).unwrap_err()
                        )
                    );
                    assert_eq!(o, before, "{}: {value:?}", key.path);
                    checked += 1;
                }
            }
        }
        assert!(checked > 0, "no inverted keys checked");
        eprintln!("[{checked} inverted group JSON refusals preserve options]");
    }

    #[test]
    fn an_inverted_leaf_without_a_remap_names_its_full_storage_path() {
        static META: KnobMeta = KnobMeta {
            key: "skip_conv_c_fusion",
            invert: true,
            ..KnobMeta::BASE
        };
        let leaf = Leaf {
            storage: "tuning.lfm2.skip_conv_c_fusion".into(),
            meta: &META,
            default: Value::Bool(false),
        };
        let mut o = EngineOptions::default();
        let before = o.clone();
        for value in ["", " \t\r\n ", "maybe", "2"] {
            let error = write_leaf(&mut o, &leaf, &Value::String(value.into()), true).unwrap_err();
            assert_eq!(
                error,
                format!(
                    "`{}`: {}",
                    leaf.storage,
                    lloom_tensor::options::parse_bool(value).unwrap_err()
                )
            );
            assert_eq!(o, before);
        }
        eprintln!("[4 unmapped inverted leaf refusals preserve options]");
    }

    #[test]
    fn the_index_builds_and_every_key_has_a_status_and_a_description() {
        let idx = Index::build().unwrap();
        for k in idx.keys.values() {
            if k.compound.is_some() {
                continue;
            }
            assert!(!k.meta().doc.is_empty(), "{} has no description", k.path);
        }
        assert!(idx.keys.len() > 250, "{} keys", idx.keys.len());
    }

    /// What `--help` and `lloom options` print names no section, issue,
    /// measured probe or source file; the notes stay in `internal`.
    #[test]
    fn no_description_carries_a_reference() {
        let is_ref = |s: &str| {
            let b = s.as_bytes();
            s.contains('§')
                || s.contains("DESIGN")
                || (0..b.len())
                    .any(|i| b[i] == b'#' && b.get(i + 1).is_some_and(u8::is_ascii_digit))
                || s.split(|c: char| c.is_whitespace() || "(),;`".contains(c))
                    .any(|w| {
                        let w = w.trim_end_matches(['.', ':']);
                        [".rs", ".md", ".metal"].iter().any(|e| w.ends_with(e))
                    })
        };
        let bad: Vec<String> = index()
            .keys
            .values()
            .map(|k| (k.path.clone(), k.meta().doc))
            .filter(|(_, d)| is_ref(d))
            .map(|(p, d)| format!("{p}: {d}"))
            .collect();
        assert!(bad.is_empty(), "{}", bad.join("\n"));
        let internal = index()
            .keys
            .values()
            .filter(|k| is_ref(k.meta().internal))
            .count();
        assert!(
            internal > 20,
            "the notes are kept: {internal} keys' internal notes cite one"
        );
    }

    #[test]
    fn a_key_and_its_storage_path_write_the_same_field() {
        let idx = index();
        let mut a = EngineOptions::default();
        idx.set(&mut a, "kernels.quant_matvec.q6_k.rows_per_simdgroup", "4")
            .unwrap();
        let mut b = EngineOptions::default();
        idx.set(&mut b, "tuning.qmv.q6k_nr0", "4").unwrap();
        assert_eq!(a, b);
        assert_eq!(a.tuning.qmv.q6k_nr0, 4);
        assert_eq!(
            idx.flat(&a)["kernels.quant_matvec.q6_k.rows_per_simdgroup"],
            4
        );
    }

    #[test]
    fn a_key_converts_its_vocabulary_and_renders_it_back() {
        let idx = index();
        let mut o = EngineOptions::default();
        idx.set(&mut o, "decode.ring.eos_wait", "event").unwrap();
        assert!(o.tuning.ring.eos_park);
        idx.set(&mut o, "memory.admission.drain", "false").unwrap();
        assert!(
            o.tuning.pool.no_admission_drain,
            "the key is the stored switch's negation"
        );
        idx.set(&mut o, "decode.host_feed.wait", "adaptive:80us")
            .unwrap();
        assert_eq!(o.tuning.ring.feed_poll, "on-demand:80");
        idx.set(&mut o, "weights.experts.fetch.wait", "sleep:2ms")
            .unwrap();
        assert_eq!(o.tuning.stream.fetch_poll_us, 2000);
        let flat = idx.flat(&o);
        assert_eq!(flat["decode.ring.eos_wait"], "event");
        assert_eq!(flat["memory.admission.drain"], false);
        assert_eq!(flat["decode.host_feed.wait"], "adaptive:80us");
        assert_eq!(flat["weights.experts.fetch.wait"], "sleep:2000us");
        assert_eq!(
            idx.flat(&EngineOptions::default())["memory.admission.drain"],
            true
        );
    }

    #[test]
    fn a_merged_key_fans_out_and_a_legacy_alias_is_reset_by_the_key() {
        let idx = index();
        let mut o = EngineOptions::default();
        idx.set(&mut o, "kernels.fusion.norm_in_gemv", "true")
            .unwrap();
        assert!(o.tuning.lfm2.norm_gemv && o.tuning.qwen4exp.hc_norm_gemv);
        idx.set(&mut o, "diagnostics.ablate", "norm,hcnorm")
            .unwrap();
        assert_eq!(o.tuning.lfm2.ablate, "norm,hcnorm");
        assert_eq!(o.tuning.qwen4exp.ablate, "norm,hcnorm");
        let mut o = EngineOptions::default();
        idx.set(&mut o, "tuning.lfm2.ablate", "norm").unwrap();
        idx.set(&mut o, "tuning.qwen4exp.ablate", "hcnorm").unwrap();
        assert_eq!(idx.flat(&o)["diagnostics.ablate"], "norm,hcnorm");
        let mut o = EngineOptions::default();
        idx.set(&mut o, "tuning.metal.nodes_per_cb", "4").unwrap();
        assert_eq!(
            idx.flat(&o)["backend.metal.command_buffer.cut_policy"],
            "nodes:4"
        );
        idx.set(
            &mut o,
            "backend.metal.command_buffer.cut_policy",
            "node-aligned",
        )
        .unwrap();
        assert_eq!(o.tuning.metal.nodes_per_cb, 0, "the key resets its alias");
        assert_eq!(
            idx.flat(&o)["backend.metal.command_buffer.cut_policy"],
            "node-aligned"
        );
    }

    #[test]
    fn a_document_in_either_shape_merges_and_the_tree_round_trips() {
        let idx = index();
        let mut o = EngineOptions::default();
        idx.merge(
            &mut o,
            &serde_json::json!({
                "$schema": "./x.json",
                "decode": {"readback": {"kind": "ring", "wait": "sleep:50us"}, "in_flight_forwards": 3},
                "tuning": {"qmv": {"q6k_nr0": 4}},
                "generation": {"stop": ["\n\n"], "logit_bias": {"3": -5}},
                "model": {"axes": {"attn": "sdpa"}},
            }),
        )
        .unwrap();
        assert_eq!(o.decode.readback.kind, crate::ReadbackKind::Ring);
        assert_eq!(o.decode.in_flight, 3);
        assert_eq!(o.tuning.qmv.q6k_nr0, 4);
        assert_eq!(o.sampling.stop, vec!["\n\n"]);
        assert_eq!(o.model.axes.get("attn").map(String::as_str), Some("sdpa"));
        let tree = idx.render(&o);
        let mut back = EngineOptions::default();
        idx.merge(&mut back, &tree).unwrap();
        assert_eq!(back, o);
        let err = idx
            .merge(
                &mut EngineOptions::default(),
                &serde_json::json!({"decode": {"nope": 1}}),
            )
            .unwrap_err();
        assert!(err.contains("unknown key `decode.nope`"), "{err}");
        let err = idx
            .merge(
                &mut EngineOptions::default(),
                &serde_json::json!({"decode": {"in_flight": 2, "in_flight_forwards": 3}}),
            )
            .unwrap_err();
        assert!(err.contains("same layer"), "{err}");
    }
}
