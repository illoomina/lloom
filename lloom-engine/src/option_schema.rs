//! The option tree's JSON Schema (2020-12), generated from the same
//! declarations as the flags and the keys (`lloom options --schema`): each
//! key's type, its default, its description, and the `x-lloom-*`
//! annotations — status, flag, env, legacy spellings, facets, flows,
//! couplings, the compiled capability it needs, each front-end's own
//! default — with the static couplings as `if/then` rules.

use crate::option_tree::{index, Key};
use crate::options::EngineOptions;
use serde_json::{json, Map, Value};

const DURATION: &str = r"\d+([uU][sS]|[mM][sS]|[sS])";

/// `pattern` (an ECMA-262 regex of ASCII alternatives) matching in any ASCII
/// case: each letter becomes its two-case class; escapes and classes are
/// copied as they are.
fn case_free(pattern: &str) -> String {
    let mut out = String::new();
    let mut chars = pattern.chars();
    let mut in_class = false;
    while let Some(c) = chars.next() {
        match c {
            '\\' => {
                out.push(c);
                if let Some(n) = chars.next() {
                    out.push(n);
                }
            }
            '[' => {
                in_class = true;
                out.push(c);
            }
            ']' => {
                in_class = false;
                out.push(c);
            }
            c if c.is_ascii_alphabetic() && !in_class => {
                out.push('[');
                out.push(c.to_ascii_lowercase());
                out.push(c.to_ascii_uppercase());
                out.push(']');
            }
            c => out.push(c),
        }
    }
    out
}

/// The value grammar (`lloom_tensor::options`) as patterns: a typed value
/// may carry surrounding whitespace, an arm and a bool spelling any case.
fn padded(body: &str) -> String {
    format!(r"^\s*({})\s*$", case_free(body))
}

fn escape(arm: &str) -> String {
    arm.chars()
        .flat_map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                vec![c]
            } else {
                vec!['\\', c]
            }
        })
        .collect()
}

/// The bool spellings, as a string pattern.
fn bool_pattern() -> String {
    let all: Vec<&str> = lloom_tensor::options::TRUE_SPELLINGS
        .iter()
        .chain(lloom_tensor::options::FALSE_SPELLINGS)
        .copied()
        .collect();
    padded(&all.join("|"))
}

/// A count, a number and `none` as the strings the grammar reads.
const COUNT: &str = r"^\s*\+?\d+\s*$";
const NUMBER: &str = r"^\s*[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?\s*$";

/// The arms a field declares (`variants`), parameterised ones included
/// (`nodes:<k>`, `ring[:<K>]`, `<count>`), as a pattern; `None` for a list
/// that is prose rather than arms.
fn variants_pattern(variants: &[&str]) -> Option<String> {
    let mut alts = Vec::new();
    for v in variants {
        if v.chars()
            .any(|c| !(c.is_ascii_alphanumeric() || "-_:<>[]µ.,".contains(c)))
        {
            return None;
        }
        if v.starts_with('<') {
            alts.push(r"\+?\d+".to_string());
            continue;
        }
        let cut = v.find([':', '[']).unwrap_or(v.len());
        let (base, rest) = v.split_at(cut);
        alts.push(match rest.chars().next() {
            None => escape(base),
            Some('[') => format!(r"{}(:[^\s]+)?", escape(base)),
            Some(_) => format!(r"{}:[^\s]+", escape(base)),
        });
    }
    Some(padded(&alts.join("|")))
}

fn arms(variants: &[&str]) -> Value {
    let alt: Vec<String> = variants.iter().map(|v| escape(v)).collect();
    json!({"anyOf": [
        {"type": "string", "enum": variants},
        {"type": "string", "pattern": padded(&alt.join("|"))}
    ]})
}

/// The shared value types.
pub fn defs() -> Value {
    json!({
        "NonnegativeNumber": {"type": "number", "minimum": 0},
        "Probability": {"type": "number", "minimum": 0, "maximum": 1},
        "Duration": {
            "type": "string",
            "pattern": format!(r"^\s*{DURATION}\s*$"),
            "description": "A time with its unit: `50us`, `1ms`, `2s`."
        },
        "Wait": {
            "type": "string",
            "pattern": format!(r"^\s*({}(:{DURATION})?)\s*$", case_free(r"block|spin|spin\-hint|wfe|inherit|(sleep|event|event\-sleep|event\-listener|adaptive)")),
            "description": "How a thread learns that something happened: `block` (the API's blocking call), `spin`, `spin-hint` (spin with a CPU hint), `wfe` (wait-for-event), `sleep[:dur]` (poll, sleeping between looks), `event[:timeout]` (park on an event), `event-sleep[:dur]` / `event-listener[:timeout]` (event variants), `adaptive[:dur]` (wfe while a consumer waits, else sleep), `inherit` (the flow's own wait). Each site takes the forms it implements (`x-lloom-wait-forms`)."
        },
        "ThreadQos": {
            "allOf": [arms(&["user-interactive", "user-initiated", "default", "utility", "background", "inherit", "off"])],
            "description": "A thread's scheduling class (macOS QoS): `inherit` = as spawned, `off` = claim nothing. Each site takes the classes it implements."
        },
        "Locality": {
            "allOf": [arms(&["device", "host"])],
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
                .map(|f| escape(f))
                .collect::<Vec<_>>()
                .join("|");
            json!({"allOf": [{"$ref": "#/$defs/Wait"}, {"pattern": format!(r"^\s*({})(:{DURATION})?\s*$", case_free(&alt))}]})
        }
        "ThreadQos" if !m.forms.is_empty() => {
            json!({"allOf": [{"$ref": "#/$defs/ThreadQos"}, arms(m.forms)]})
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
                    b = arms(&vs);
                } else {
                    if let Some(p) = variants_pattern(&vs) {
                        b["pattern"] = json!(p);
                    }
                    b["examples"] = json!(m.variants);
                }
            }
            b
        }
        _ if base_type(ty)["type"] == "number" => base_type(ty),
        other => json!({"$ref": format!("#/$defs/{other}")}),
    };
    // A key whose vocabulary moved to a shared type still takes the stored
    // spellings it always took (`interactive`, `synchronize`).
    if m.schema != "" && !m.variants.is_empty() {
        if let Some(p) = variants_pattern(m.variants) {
            s = json!({"anyOf": [s, {"type": "string", "pattern": p}]});
        }
    }
    // What the grammar also reads besides the JSON type: a bool's spellings
    // (and 0/1), a count's or a number's digits as a string.
    if m.is_bool || ty == "bool" {
        s = json!({"anyOf": [s, {"type": "boolean"}, {"enum": [0, 1]}, {"type": "string", "pattern": bool_pattern()}]});
    } else if base_type(ty)["type"] == "integer" {
        s = json!({"anyOf": [s, {"type": "integer", "minimum": 0}, {"type": "string", "pattern": COUNT}]});
    } else if base_type(ty)["type"] == "number" {
        s = json!({"anyOf": [s, {"type": "number"}, {"type": "string", "pattern": NUMBER}]});
    }
    // The derive applies these exact named inputs before the stored type's
    // parser (e.g. named methods stored as bool). An annotation alone does
    // not admit them; preserve the stored grammar alongside the input names.
    if !m.values.is_empty() {
        let names: Vec<&str> = m.values.iter().map(|(input, _)| *input).collect();
        s = json!({"anyOf": [s, {"type": "string", "enum": names}]});
    }
    // `null` returns any key to its default; `none` unsets an optional one.
    let mut alts = vec![s, json!({"type": "null"})];
    if optional || default.is_null() {
        alts.push(json!({"type": "string", "pattern": padded("none")}));
    }
    json!({"anyOf": alts})
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
    if !m.needs.is_empty() {
        let mut needs = Map::new();
        needs.insert("capability".into(), json!(m.needs));
        if !m.needs_part.is_empty() {
            needs.insert("part".into(), json!(m.needs_part));
        }
        s.insert("x-lloom-needs".into(), Value::Object(needs));
    }
    let frontends: Vec<Value> = crate::frontend::rows_of(&key.path)
        .map(|r| {
            let typed = |v: &str| {
                let mut o = EngineOptions::default();
                index().set(&mut o, &key.path, v).expect("a row is a value of its key");
                index().flat(&o).remove(&key.path).unwrap_or(Value::Null)
            };
            let mut e = Map::new();
            e.insert("frontend".into(), json!(r.frontend.name()));
            e.insert("value".into(), typed(crate::frontend::value(r)));
            if r.when != crate::frontend::When::Always {
                e.insert("when".into(), json!(r.when.describe()));
            }
            if let Some(x) = crate::frontend::exception(r) {
                e.insert(
                    "change".into(),
                    json!({"decision": x.decision, "before": typed(x.before), "after": typed(x.after), "landed": x.landed}),
                );
            }
            Value::Object(e)
        })
        .collect();
    if !frontends.is_empty() {
        s.insert("x-lloom-frontend-defaults".into(), Value::Array(frontends));
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

/// Numeric domains belong to full-config checking, not the leaf spelling
/// grammar: parsing a number and accepting a sampling request are separate.
/// Numeric strings retain their loader grammar here; their domain projection
/// also needs the loader's represented-f64 rounding, not a decimal sign test.
fn numeric_domain_rules() -> Vec<Value> {
    index().keys.values().filter_map(|key| {
        let m = key.meta();
        let ty = m.ty.strip_prefix("Option<").and_then(|t| t.strip_suffix('>')).unwrap_or(m.ty);
        if m.schema.is_empty() || base_type(ty)["type"] != "number" { return None; }
        let mut node = json!({"if": {"type": "number"}, "then": {"$ref": format!("#/$defs/{}", m.schema)}});
        for part in key.path.split('.').rev() {
            node = json!({"properties": {part: node}});
        }
        Some(node)
    }).collect()
}

/// The readback forms selecting a kind, through the same preset parser as
/// the option loader. Couplings must not apply object keywords to strings.
pub(crate) fn readback_kind(kind: crate::ReadbackKind) -> Value {
    let spellings: Vec<&str> = crate::ReadbackConfig::SPELLINGS
        .iter()
        .copied()
        .filter(|s| {
            let mut cfg = crate::ReadbackConfig::default();
            crate::ReadbackConfig::apply_spelling(&mut cfg, s).is_ok() && cfg.kind == kind
        })
        .collect();
    let name = kind.to_string();
    json!({"anyOf": [
        {"type": "object", "properties": {"kind": arms(&[name.as_str()])}, "required": ["kind"]},
        {"type": "string", "enum": spellings}
    ]})
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
    root["allOf"]
        .as_array_mut()
        .expect("static rules are an array")
        .extend(numeric_domain_rules());
    root
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numeric_domains_are_field_attached_config_rules_not_a_second_leaf_parser() {
        let schema = schema();
        let rules = schema["allOf"].as_array().unwrap();
        for (path, domain) in [
            ("generation.temperature", "NonnegativeNumber"),
            ("generation.top_p", "Probability"),
        ] {
            let key = &index().keys[path];
            assert_eq!(key.meta().schema, domain);
            let field = path.rsplit('.').next().unwrap();
            let expected = json!({"properties": {"generation": {"properties": {
                field: {"if": {"type": "number"},
                    "then": {"$ref": format!("#/$defs/{domain}")}}
            }}}});
            assert!(
                rules.contains(&expected),
                "{path}: missing full-config domain"
            );
            let default = Value::Null;
            assert_eq!(
                value_schema(key, &default)["anyOf"][0]["anyOf"][0],
                json!({"type": "number"}),
                "leaf grammar still parses numbers before full-config checking"
            );
        }
        assert_eq!(
            schema["$defs"]["NonnegativeNumber"],
            json!({"type": "number", "minimum": 0})
        );
        assert_eq!(
            schema["$defs"]["Probability"],
            json!({"type": "number", "minimum": 0, "maximum": 1})
        );
    }

    #[test]
    fn declared_value_mappings_add_exact_schema_names() {
        let schema = schema();
        let mut mapped = 0;
        for key in index().keys.values().filter(|key| !key.meta().values.is_empty()) {
            let node = key.path.split('.').fold(&schema, |node, part| {
                &node["properties"][part]
            });
            let names: Vec<&str> = key.meta().values.iter().map(|(input, _)| *input).collect();
            assert_eq!(
                node["anyOf"][0]["anyOf"][1],
                json!({"type": "string", "enum": names}),
                "{}: declared input names must be validation, not only annotations",
                key.path
            );
            assert!(node["anyOf"][0]["anyOf"][0].is_object());
            mapped += 1;
        }
        assert!(mapped > 0, "control must reach a real mapped declaration");
    }

    #[test]
    fn readback_coupling_forms_follow_the_actual_preset_parser() {
        let schema = schema();
        let predicate = &schema["allOf"][0]["if"]["properties"]["decode"]["properties"]["readback"];
        assert_eq!(predicate["anyOf"][0]["type"], "object");
        assert_eq!(predicate["anyOf"][1]["type"], "string");
        let strings = predicate["anyOf"][1]["enum"].as_array().unwrap();
        for spelling in crate::ReadbackConfig::SPELLINGS {
            let mut cfg = crate::ReadbackConfig::default();
            crate::ReadbackConfig::apply_spelling(&mut cfg, spelling).unwrap();
            assert_eq!(
                strings.contains(&json!(spelling)),
                cfg.kind == crate::ReadbackKind::Ring,
                "{spelling}"
            );
        }
        assert_eq!(
            schema["allOf"][1]["if"]["properties"]["decode"]["properties"]["readback"]["type"],
            "object"
        );
    }

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
