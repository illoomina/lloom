//! The option tree's JSON Schema (2020-12), generated from the same
//! declarations as the flags and the keys (`lloom options --schema`): each
//! key's type, its default, its description, and the `x-lloom-*`
//! annotations — status, flag, env, legacy spellings, facets, flows,
//! couplings — with the static couplings as `if/then` rules.

use crate::option_tree::{index, Key};
use crate::options::EngineOptions;
use serde_json::{json, Map, Value};

const DURATION: &str = r"\d+(us|ms|s)";

/// The shared value types.
pub fn defs() -> Value {
    json!({
        "Duration": {
            "type": "string",
            "pattern": format!("^{DURATION}$"),
            "description": "A time with its unit: `50us`, `1ms`, `2s`."
        },
        "Wait": {
            "type": "string",
            "pattern": format!("^(block|spin|spin-hint|wfe|inherit|(sleep|event|event-sleep|event-listener|adaptive)(:{DURATION})?)$"),
            "description": "How a thread learns that something happened: `block` (the API's blocking call), `spin`, `spin-hint` (spin with a CPU hint), `wfe` (wait-for-event), `sleep[:dur]` (poll, sleeping between looks), `event[:timeout]` (park on an event), `event-sleep[:dur]` / `event-listener[:timeout]` (event variants), `adaptive[:dur]` (wfe while a consumer waits, else sleep), `inherit` (the flow's own wait). Each site takes the forms it implements (`x-lloom-wait-forms`)."
        },
        "ThreadQos": {
            "type": "string",
            "enum": ["user-interactive", "user-initiated", "default", "utility", "background", "inherit", "off"],
            "description": "A thread's scheduling class (macOS QoS): `inherit` = as spawned, `off` = claim nothing. Each site takes the classes it implements."
        },
        "Locality": {
            "type": "string",
            "enum": ["device", "host"],
            "description": "Where a decision is made: on the GPU, or on the host after a readback."
        }
    })
}

fn literal(v: &str) -> bool {
    !v.is_empty()
        && v.chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || "-_.x".contains(c))
}

fn base_type(ty: &str) -> Value {
    match ty {
        "bool" => json!({"type": "boolean"}),
        "usize" | "u64" | "u32" | "u16" | "u8" => json!({"type": "integer", "minimum": 0}),
        "f64" | "f32" => json!({"type": "number"}),
        t if t.starts_with("Vec<") => json!({"type": "array", "items": {"type": "string"}}),
        t if t.contains("BTreeMap<") => {
            json!({"type": "object", "additionalProperties": {"type": "number"}})
        }
        _ => json!({"type": "string"}),
    }
}

/// The JSON Schema of one key's value.
fn value_schema(key: &Key, default: &Value) -> Value {
    let m = key.meta();
    let (ty, optional) = match m
        .ty
        .strip_prefix("Option<")
        .and_then(|t| t.strip_suffix('>'))
    {
        Some(inner) => (inner, true),
        None => (m.ty, false),
    };
    let mut s = match m.schema {
        "Wait" => {
            let alt = m
                .forms
                .iter()
                .map(|f| f.replace('-', "\\-"))
                .collect::<Vec<_>>()
                .join("|");
            json!({"allOf": [{"$ref": "#/$defs/Wait"}, {"pattern": format!("^({alt})(:{DURATION})?$")}]})
        }
        "ThreadQos" if !m.forms.is_empty() => {
            json!({"allOf": [{"$ref": "#/$defs/ThreadQos"}, {"enum": m.forms}]})
        }
        "" => {
            let mut b = if m.is_bool {
                json!({"type": "boolean"})
            } else {
                base_type(ty)
            };
            if b["type"] == "string" && !m.variants.is_empty() {
                let vs: Vec<&str> = m.variants.to_vec();
                if vs.iter().all(|v| literal(v)) {
                    b["enum"] = json!(vs);
                } else {
                    b["examples"] = json!(m.variants);
                }
            }
            b
        }
        other => json!({"$ref": format!("#/$defs/{other}")}),
    };
    if optional || default.is_null() {
        s = json!({"anyOf": [s, {"type": "null"}]});
    }
    s
}

fn annotate(key: &Key, default: &Value, s: &mut Map<String, Value>) {
    let m = key.meta();
    s.insert("default".into(), default.clone());
    s.insert("description".into(), json!(m.doc));
    s.insert("x-lloom-status".into(), json!(m.status.name()));
    if let Some(c) = key.cli() {
        s.insert("x-lloom-flag".into(), json!(format!("--{c}")));
        if let Some(short) = key.targets.iter().find_map(|l| l.meta.short) {
            s.insert("x-lloom-short".into(), json!(format!("-{short}")));
        }
    }
    s.insert("x-lloom-env".into(), json!(key.env()));
    let mut legacy = Vec::new();
    for l in key.targets.iter().chain(&key.legacy) {
        let mut e = Map::new();
        if l.meta.v0 && l.storage != key.path {
            e.insert("key".into(), json!(l.storage));
        }
        let flag = l.meta.flag;
        if !flag.is_empty() && Some(flag) != key.cli() {
            e.insert("flag".into(), json!(format!("--{flag}")));
        }
        if let Some(env) = l.meta.env {
            e.insert("env".into(), json!(env));
        }
        if !e.is_empty() {
            legacy.push(Value::Object(e));
        }
        for k in l.meta.legacy_keys {
            legacy.push(json!({"key": k}));
        }
        for a in l.meta.aliases {
            legacy.push(json!({"flag": format!("--{a}")}));
        }
        for (sw, v) in l.meta.switches {
            legacy.push(json!({"flag": format!("--{sw}"), "value": v}));
        }
    }
    if !legacy.is_empty() {
        s.insert("x-lloom-legacy".into(), Value::Array(legacy));
    }
    if !m.facets.is_empty() {
        s.insert("x-lloom-facets".into(), json!(m.facets));
    }
    if !m.flows.is_empty() {
        s.insert("x-lloom-flows".into(), json!(m.flows));
    }
    let couplings = crate::couplings::of(&key.path);
    if !couplings.is_empty() {
        s.insert("x-lloom-couplings".into(), json!(couplings));
    }
    if m.schema == "Wait" {
        s.insert("x-lloom-wait-forms".into(), json!(m.forms));
    }
    if !m.values.is_empty() {
        let aliases: Map<String, Value> = m
            .values
            .iter()
            .map(|(k, st)| (st.to_string(), json!(k)))
            .collect();
        s.insert("x-lloom-value-aliases".into(), Value::Object(aliases));
    }
    if key.targets.len() > 1 {
        s.insert(
            "x-lloom-merged".into(),
            json!(key
                .targets
                .iter()
                .map(|t| t.storage.clone())
                .collect::<Vec<_>>()),
        );
    }
}

fn group_node() -> Value {
    json!({"type": "object", "additionalProperties": false, "properties": {}})
}

fn put(root: &mut Value, path: &str, leaf: Value) {
    let segs: Vec<&str> = path.split('.').collect();
    let mut node = root;
    for seg in &segs[..segs.len() - 1] {
        node = node["properties"]
            .as_object_mut()
            .expect("a group")
            .entry(seg.to_string())
            .or_insert_with(group_node);
    }
    node["properties"]
        .as_object_mut()
        .expect("a group")
        .insert(segs[segs.len() - 1].to_string(), leaf);
}

/// The schema of the whole tree.
pub fn schema() -> Value {
    let idx = index();
    let defaults = idx.flat(&EngineOptions::default());
    let mut root = json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://lloom.dev/schema/options.v1.json",
        "title": "lloom options",
        "type": "object",
        "additionalProperties": false,
        "properties": {"$schema": {"type": "string"}},
    });
    for key in idx.keys.values() {
        if key.compound.is_some() {
            continue;
        }
        let default = defaults.get(&key.path).cloned().unwrap_or(Value::Null);
        let mut s = value_schema(key, &default);
        let mut obj = match s.take() {
            Value::Object(m) => m,
            _ => unreachable!(),
        };
        annotate(key, &default, &mut obj);
        put(&mut root, &key.path, Value::Object(obj));
    }
    // A group's shorthand is also accepted as a string in the group's place.
    for key in idx.keys.values().filter(|k| k.compound.is_some()) {
        let segs: Vec<&str> = key.path.split('.').collect();
        let mut node = &mut root;
        for seg in &segs[..segs.len() - 1] {
            node = &mut node["properties"][*seg];
        }
        let last = segs[segs.len() - 1];
        let group = node["properties"][last].take();
        let m = key.meta();
        node["properties"][last] = json!({
            "anyOf": [group, {"type": "string", "enum": crate::ReadbackConfig::SPELLINGS}],
            "description": format!("{} A string is the shorthand for the group's keys.", m.doc),
            "x-lloom-status": m.status.name(),
            "x-lloom-flag": format!("--{}", m.cli),
        });
    }
    root["$defs"] = defs();
    root["x-lloom-couplings"] = crate::couplings::table();
    root["allOf"] = crate::couplings::static_rules();
    root
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every key of the index has a node in the schema, at its path, and
    /// nothing else does.
    #[test]
    fn the_schema_has_exactly_the_trees_keys() {
        fn leaves(v: &Value, prefix: &str, out: &mut Vec<String>) {
            let Some(props) = v.get("properties").and_then(Value::as_object) else {
                return;
            };
            for (k, child) in props {
                if prefix.is_empty() && k == "$schema" {
                    continue;
                }
                let p = if prefix.is_empty() {
                    k.clone()
                } else {
                    format!("{prefix}.{k}")
                };
                let group = child
                    .get("anyOf")
                    .and_then(|a| a.get(0))
                    .filter(|g| g.get("properties").is_some())
                    .unwrap_or(child);
                if group.get("properties").is_some() {
                    leaves(group, &p, out);
                } else {
                    out.push(p);
                }
            }
        }
        let s = schema();
        let mut got = Vec::new();
        leaves(&s, "", &mut got);
        got.sort();
        let want: Vec<String> = index()
            .keys
            .values()
            .filter(|k| k.compound.is_none())
            .map(|k| k.path.clone())
            .collect();
        assert_eq!(got, want);
    }
}
