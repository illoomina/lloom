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
    format!("^{TRIM_SPACE}*({}){TRIM_SPACE}*$", case_free(body))
}

/// Quote a literal arm outside character classes for both ECMA modes.
fn escape(arm: &str) -> String {
    arm.chars()
        .flat_map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '_' | '-') {
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
const NUMBER: &str = r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?";

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

/// Decimal strings in 0..=maximum, including arbitrarily many leading zeros.
/// Prefix branches differ at their first digit below the upper bound.
fn bounded_decimal(maximum: u64) -> String {
    count_decimal(maximum, false)
}

fn count_decimal(maximum: u64, positive: bool) -> String {
    if positive && maximum == 0 {
        return r"[^\s\S]".into();
    }
    let bound = maximum.to_string();
    let width = bound.len();
    let mut alts = if positive {
        Vec::new()
    } else {
        vec!["0".to_string()]
    };
    if width > 1 {
        alts.push(format!(r"[1-9][0-9]{{0,{}}}", width - 2));
    }
    for (i, digit) in bound.bytes().enumerate() {
        let first = if i == 0 { b'1' } else { b'0' };
        if digit > first {
            alts.push(format!(
                "{}[{}-{}][0-9]{{{}}}",
                &bound[..i],
                char::from(first),
                char::from(digit - 1),
                width - i - 1,
            ));
        }
    }
    alts.push(bound);
    format!("0*({})", alts.join("|"))
}

/// Rust trim's whitespace set, written explicitly for both Rust regex and
/// ECMA-262 (whose \s includes FEFF and excludes NEL).
const TRIM_SPACE: &str = "[\u{0009}-\u{000D}\u{0020}\u{0085}\u{00A0}\u{1680}\u{2000}-\u{200A}\u{2028}\u{2029}\u{202F}\u{205F}\u{3000}]";

fn readback_shorthand_schema() -> Value {
    let variants = crate::ReadbackConfig::SPELLINGS;
    let alts: Vec<String> = variants
        .iter()
        .map(|s| escape(s).replace(r"\-", "-"))
        .collect();
    json!({"anyOf": [
        {"type": "string", "enum": variants},
        {"type": "string", "pattern": format!("^{TRIM_SPACE}*({}){TRIM_SPACE}*$", case_free(&alts.join("|")))}
    ]})
}

fn wait_schema(grammar: &lloom_tensor::options::wait::WaitGrammar) -> Value {
    let literal = |s: &str| escape(s).replace(r"\-", "-");
    let mut alts: Vec<String> = grammar.literals.iter().map(|s| literal(s)).collect();
    alts.extend(grammar.literal_aliases.iter().map(|(s, _)| literal(s)));
    if let Some(count) = grammar.count {
        alts.push(format!(
            r"\+?{}",
            count_decimal(count.domain.maximum(), true)
        ));
    }
    for arm in grammar.parameter_arms() {
        let spelling = literal(arm.spelling);
        let maximum = grammar.count.map_or(arm.parameter.maximum(), |c| {
            c.domain.maximum().min(arm.parameter.maximum())
        });
        let digits = |maximum| count_decimal(maximum, grammar.count.is_some());
        let raw = format!(
            "{}{}{}",
            if arm.parameter.raw_trim() {
                format!("{TRIM_SPACE}*")
            } else {
                String::new()
            },
            if arm.parameter.raw_plus() { r"\+?" } else { "" },
            digits(maximum),
        );
        let mut parameters = vec![raw];
        for unit in arm.parameter.units() {
            let quantum = arm.parameter.duration_quantum(*unit);
            let digits = if quantum == 1 {
                digits(maximum / unit.scale())
            } else {
                format!(
                    "(0+|{}{})",
                    bounded_decimal(maximum / unit.scale() / quantum),
                    "0".repeat(quantum.ilog10() as usize),
                )
            };
            parameters.push(format!(
                r"{TRIM_SPACE}*\+?{}{}",
                digits,
                literal(unit.suffix()),
            ));
        }
        alts.push(spelling.clone());
        alts.push(format!("{spelling}:({})", parameters.join("|")));
    }
    json!({"type": "string", "pattern": format!(
        "^{TRIM_SPACE}*({}){TRIM_SPACE}*$", case_free(&alts.join("|"))
    )})
}

/// The shared value types.
pub fn defs() -> Value {
    json!({
        "NonnegativeNumber": {"type": "number", "minimum": 0},
        "Probability": {"type": "number", "minimum": 0, "maximum": 1},
        "Duration": {
            "type": "string",
            "pattern": format!(r"^{TRIM_SPACE}*{DURATION}{TRIM_SPACE}*$"),
            "description": "A time with its unit: `50us`, `1ms`, `2s`."
        },
        "Wait": {
            "type": "string",
            "pattern": format!(r"^{TRIM_SPACE}*({}(:{DURATION})?){TRIM_SPACE}*$", case_free(r"block|spin|spin-hint|wfe|inherit|(sleep|event|event-sleep|event-listener|adaptive)")),
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

/// Storage-width bounds, from the same primitive type recorded by the derive.
/// `usize` follows the schema-producing binary's target, not the reader's host.
fn unsigned_width(ty: &str) -> Option<(u32, u64)> {
    Some(match ty {
        "u8" => (u8::BITS, u64::from(u8::MAX)),
        "u16" => (u16::BITS, u64::from(u16::MAX)),
        "u32" => (u32::BITS, u64::from(u32::MAX)),
        "u64" => (u64::BITS, u64::MAX),
        "usize" => (
            usize::BITS,
            u64::try_from(usize::MAX).expect("supported target width"),
        ),
        _ => return None,
    })
}

fn base_type(ty: &str) -> Value {
    if let Some((_, maximum)) = unsigned_width(ty) {
        return json!({"type": "integer", "minimum": 0, "maximum": maximum});
    }
    match ty {
        "bool" => json!({"type": "boolean"}),
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
    let normalized_bool = m.normalize_values && (m.is_bool || ty == "bool");
    let mut s = if normalized_bool {
        let aliases = m
            .values
            .iter()
            .map(|(key, _)| escape(key).replace(r"\-", "-"))
            .collect::<Vec<_>>();
        let spellings = lloom_tensor::options::TRUE_SPELLINGS
            .iter()
            .chain(lloom_tensor::options::FALSE_SPELLINGS)
            .map(|s| s.to_string())
            .chain(aliases)
            .collect::<Vec<_>>();
        json!({"type": "string", "pattern": format!("^{TRIM_SPACE}*({}){TRIM_SPACE}*$", case_free(&spellings.join("|")))})
    } else {
        match (m.grammar, m.schema) {
            (Some(grammar), _) => wait_schema(grammar),
            (None, "Wait") => {
                let alt = m
                    .forms
                    .iter()
                    .map(|f| escape(f))
                    .collect::<Vec<_>>()
                    .join("|");
                json!({"allOf": [{"$ref": "#/$defs/Wait"}, {"pattern": format!(r"^{TRIM_SPACE}*({})(:{DURATION})?{TRIM_SPACE}*$", case_free(&alt))}]})
            }
            (None, "ThreadQos") if !m.forms.is_empty() => {
                json!({"allOf": [{"$ref": "#/$defs/ThreadQos"}, arms(m.forms)]})
            }
            (None, "") => {
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
            // A numeric domain is a full-config rule (`numeric_domain_rules`),
            // not a leaf spelling: the leaf keeps the number grammar.
            (None, _) if base_type(ty)["type"] == "number" => base_type(ty),
            (None, other) => json!({"$ref": format!("#/$defs/{other}")}),
        }
    };
    // A key whose vocabulary moved to a shared type still takes the stored
    // spellings it always took (`interactive`, `synchronize`).
    if !normalized_bool && m.grammar.is_none() && !m.schema.is_empty() && !m.variants.is_empty() {
        if let Some(p) = variants_pattern(m.variants) {
            s = json!({"anyOf": [s, {"type": "string", "pattern": p}]});
        }
    }
    // What the grammar also reads besides the JSON type: a bool's spellings
    // (and 0/1), a count's or a number's digits as a string.
    if let Some(count) = m.grammar.and_then(|g| g.count) {
        s = json!({"anyOf": [s, {"type": "integer", "minimum": count.domain.minimum(), "maximum": count.domain.maximum()}]});
    } else if normalized_bool {
        s = json!({"anyOf": [s, {"type": "boolean"}, {"enum": [0, 1]}]});
    } else if m.is_bool || ty == "bool" {
        s = json!({"anyOf": [s, {"type": "boolean"}, {"enum": [0, 1]}, {"type": "string", "pattern": bool_pattern()}]});
    } else if let Some((_, maximum)) = unsigned_width(ty) {
        s = json!({"anyOf": [s, base_type(ty), {"type": "string", "pattern": padded(&format!(r"\+?{}", bounded_decimal(maximum)))}]});
    } else if base_type(ty)["type"] == "number" {
        s = json!({"anyOf": [s, {"type": "number"}, {"type": "string", "pattern": padded(NUMBER)}]});
    }
    // The derive applies these exact named inputs before the stored type's
    // parser (e.g. named methods stored as bool). An annotation alone does
    // not admit them; preserve the stored grammar alongside the input names.
    // A normalized bool already admits its aliases in its own pattern.
    if !m.values.is_empty() && !normalized_bool {
        let names: Vec<&str> = m.values.iter().map(|(input, _)| *input).collect();
        s = json!({"anyOf": [s, {"type": "string", "enum": names}]});
    }
    // `null` returns any key to its default; `none` unsets an optional one.
    let mut alts = vec![s, json!({"type": "null"})];
    if optional || default.is_null() {
        alts.push(json!({"type": "string", "pattern": padded("none")}));
    }
    let mut node = json!({"anyOf": alts});
    if let Some((bits, maximum)) = unsigned_width(ty) {
        node["x-lloom-unsigned-domain"] = json!({
            "storage-type": ty, "bits": bits, "minimum": 0, "maximum": maximum,
            "target-dependent": ty == "usize",
            "scope": "storage width; a field-specific parser may impose further constraints",
            "native-representation": if m.grammar.and_then(|g| g.count).is_some() {
                "field-attached count domain checks integral represented values before scalarization"
            } else {
                "JSON Schema integer is mathematical; generic unsigned ingress currently requires serde_json integer representation, not floating-point 1.0"
            }
        });
    }
    node
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
    if let Some(grammar) = m.grammar {
        if let Some(count) = grammar.count {
            s.insert("x-lloom-native-count".into(), json!({"minimum": count.domain.minimum(), "maximum": count.domain.maximum(), "representation": "serde_json::Number; integral represented value"}));
        }
        if let Some(parser) = grammar.parser {
            s.insert("x-lloom-wait-parser".into(), json!(parser));
            s.insert("x-lloom-wait-parameters".into(), json!(grammar.parameter_arms().map(|arm| {
                json!({"spelling": arm.spelling, "stored": arm.stored,
                    "raw-maximum": arm.parameter.maximum(),
                    "raw-plus": arm.parameter.raw_plus(), "raw-trim": arm.parameter.raw_trim(),
                    "duration-units": arm.parameter.units().iter().map(|u| u.suffix()).collect::<Vec<_>>(),
                    "duration-domains": arm.parameter.units().iter().map(|u| json!({
                        "unit": u.suffix(), "checked-microsecond-scale": u.scale(),
                        "maximum-count": arm.parameter.maximum() / u.scale(),
                        "count-multiple": arm.parameter.duration_quantum(*u),
                    })).collect::<Vec<_>>()})
            }).collect::<Vec<_>>()));
        }
    }
    if !m.values.is_empty() {
        if m.normalize_values {
            s.insert(
                "x-lloom-value-alias-normalization".into(),
                json!("ASCII case-insensitive; Rust outer trim"),
            );
        }
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
            "anyOf": [group, readback_shorthand_schema()],
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
        // A normalized bool admits its aliases through its normalization
        // pattern (`option_acceptance_tests` holds that schema).
        for key in index()
            .keys
            .values()
            .filter(|key| !key.meta().values.is_empty() && !key.meta().normalize_values)
        {
            let node = key
                .path
                .split('.')
                .fold(&schema, |node, part| &node["properties"][part]);
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

    /// Exercise the real derive/parser for every supported storage width,
    /// without adding synthetic options to the product registry.
    #[derive(
        lloom_macros::Options,
        serde::Serialize,
        serde::Deserialize,
        Default,
        Debug,
        Clone,
        PartialEq,
    )]
    struct UnsignedProbe {
        #[opt]
        tiny: u8,
        #[opt]
        short: u16,
        #[opt]
        word: u32,
        #[opt]
        wide: u64,
        #[opt]
        native: usize,
        #[opt]
        optional: Option<u16>,
    }

    fn probe_key(meta: &'static lloom_tensor::options::KnobMeta) -> Key {
        Key {
            path: meta.key.into(),
            targets: vec![crate::option_tree::Leaf {
                storage: meta.key.into(),
                meta,
                default: json!(0),
            }],
            legacy: vec![],
            compound: None,
            own: None,
        }
    }

    #[test]
    fn outside_class_literal_arms_keep_hyphens_and_regex_boundaries() {
        use crate::option_grammar::tests::{schema_node, schema_valid};
        for (literal, other) in [
            ("user-interactive", "userXinteractive"),
            ("per-token", "perXtoken"),
            ("tile[1].x", "tile1x"),
            ("a|b", "a"),
        ] {
            let pattern = padded(&escape(literal));
            assert!(
                !pattern.contains(r"\-"),
                "outside-class identity escape: {pattern}"
            );
            let regex = regex::Regex::new(&pattern).unwrap();
            assert!(regex.is_match(literal), "{literal}: {pattern}");
            assert!(!regex.is_match(other), "{other}: {pattern}");
        }
        let schema = schema();
        let idx = index();
        for (path, value) in [
            ("context.kv.layout", "in-place"),
            ("decode.host_feed.qos", "user-interactive"),
            ("executor.replay.barriers", "skip-replayed"),
            ("session.missing_reasoning", "as-sent"),
        ] {
            assert!(
                schema_valid(schema_node(&schema, path), &json!(value), &schema["$defs"]),
                "{path}: {value}"
            );
            let mut options = EngineOptions::default();
            idx.set(&mut options, path, value).unwrap();
        }
    }

    #[test]
    fn shared_wait_definition_keeps_literal_hyphens_and_parameter_boundaries() {
        let defs = defs();
        let pattern = defs["Wait"]["pattern"].as_str().unwrap();
        assert!(
            !pattern.contains(r"\-"),
            "outside-class identity escape: {pattern}"
        );
        let regex = regex::Regex::new(pattern).unwrap();
        for value in [
            "spin-hint",
            "event-sleep",
            "event-listener:50us",
            " EVENT-LISTENER:50US ",
        ] {
            assert!(regex.is_match(value), "{value}: {pattern}");
        }
        for value in [
            "spinXhint",
            "eventXsleep",
            "eventXlistener:50us",
            "event-listener:bad",
        ] {
            assert!(!regex.is_match(value), "{value}: {pattern}");
        }
    }

    #[test]
    fn unsigned_widths_bound_native_and_decimal_string_domains() {
        use crate::option_grammar::tests::schema_valid;
        let defs = defs();
        let mut checked = 0;
        for meta in UnsignedProbe::META.iter().filter(|m| m.key != "optional") {
            let (bits, maximum) = unsigned_width(meta.ty).expect("unsigned probe");
            let node = value_schema(&probe_key(meta), &json!(0));
            assert_eq!(node["x-lloom-unsigned-domain"]["bits"], bits);
            assert_eq!(node["x-lloom-unsigned-domain"]["maximum"], maximum);
            for n in [0, 1, maximum - 1, maximum] {
                let value = json!(n);
                assert!(schema_valid(&node, &value, &defs), "{}: {value}", meta.ty);
                let mut probe = UnsignedProbe::default();
                probe.merge_json(&json!({meta.key: value})).unwrap();
                assert_eq!(serde_json::to_value(&probe).unwrap()[meta.key], n);
                for text in [
                    n.to_string(),
                    format!("+{n}"),
                    format!("\u{0085}+000000{n}\u{0085}"),
                ] {
                    assert!(
                        schema_valid(&node, &json!(text), &defs),
                        "{}: {text:?}",
                        meta.ty
                    );
                    probe.set(meta.key, &text).unwrap();
                    assert_eq!(serde_json::to_value(&probe).unwrap()[meta.key], n);
                }
            }
            // u128 here is only a fixture constructor; no u128 product type
            // or floating conversion is used to represent the upper bound.
            let above = (u128::from(maximum) + 1).to_string();
            for text in [
                above.clone(),
                format!("+000{above}"),
                "-1".into(),
                "1.0".into(),
                "1e0".into(),
                "\u{feff}1".into(),
            ] {
                let before = UnsignedProbe::default();
                let mut probe = before.clone();
                assert!(
                    !schema_valid(&node, &json!(text), &defs),
                    "{}: {text:?}",
                    meta.ty
                );
                assert!(probe.set(meta.key, &text).is_err(), "{}: {text:?}", meta.ty);
                assert_eq!(probe, before, "refusal changed {}", meta.key);
            }
            let above = if maximum < u64::MAX {
                json!(maximum + 1)
            } else {
                json!(2.0_f64.powi(64))
            };
            for value in [json!(-1), above, json!(1.5)] {
                let before = UnsignedProbe::default();
                let mut probe = before.clone();
                assert!(!schema_valid(&node, &value, &defs), "{}: {value}", meta.ty);
                assert!(probe.merge_json(&json!({meta.key: value})).is_err());
                assert_eq!(probe, before);
            }
            checked += 1;
        }
        assert_eq!(checked, 5);
    }

    #[test]
    fn optional_unsigned_width_preserves_unset_and_maximum() {
        use crate::option_grammar::tests::schema_valid;
        let meta = UnsignedProbe::META
            .iter()
            .find(|m| m.key == "optional")
            .unwrap();
        let node = value_schema(&probe_key(meta), &Value::Null);
        for value in [
            Value::Null,
            json!("none"),
            json!(u16::MAX),
            json!("+00065535"),
        ] {
            assert!(schema_valid(&node, &value, &defs()));
        }
        for value in [json!(65536), json!("65536")] {
            assert!(!schema_valid(&node, &value, &defs()));
        }
        let mut probe = UnsignedProbe::default();
        probe.set("optional", "+00065535").unwrap();
        assert_eq!(probe.optional, Some(u16::MAX));
        let before = probe.clone();
        assert!(probe.set("optional", "65536").is_err());
        assert_eq!(probe, before);
        probe.set("optional", "none").unwrap();
        assert_eq!(probe.optional, None);
    }

    #[test]
    fn unsigned_schema_preserves_integer_precision_and_documents_native_float_gap() {
        use crate::option_grammar::tests::schema_valid;
        let meta = UnsignedProbe::META
            .iter()
            .find(|m| m.key == "wide")
            .unwrap();
        let node = value_schema(&probe_key(meta), &json!(0));
        assert!(schema_valid(&node, &json!(u64::MAX), &defs()));
        // Casting u64::MAX to f64 rounds UP to 2^64. Never round the
        // schema's maximum to f64 and accidentally admit that value.
        let rounded = u64::MAX as f64;
        assert_eq!(rounded, 2.0_f64.powi(64));
        assert!(!schema_valid(&node, &json!(rounded), &defs()));
        let below = f64::from_bits(rounded.to_bits() - 1);
        assert!(schema_valid(&node, &json!(below), &defs()));
        // Mathematical JSON Schema integer cannot express the current
        // generic ingress distinction between represented 1 and 1.0.
        assert!(schema_valid(&node, &json!(1.0), &defs()));
        let mut probe = UnsignedProbe::default();
        assert!(probe.merge_json(&json!({"wide": 1.0})).is_err());
        assert_eq!(probe, UnsignedProbe::default());
        assert!(node["x-lloom-unsigned-domain"]["native-representation"]
            .as_str()
            .unwrap()
            .contains("1.0"));
    }

    #[test]
    fn actual_server_port_schema_and_ingress_keep_u16_boundary() {
        use crate::option_grammar::tests::{schema_node, schema_valid};
        let schema = schema();
        let node = schema_node(&schema, "server.port");
        let idx = index();
        for value in [json!(65535), json!("+00065535")] {
            assert!(schema_valid(node, &value, &schema["$defs"]));
            let mut options = EngineOptions::default();
            idx.merge(&mut options, &json!({"server": {"port": value}}))
                .unwrap();
            assert_eq!(idx.flat(&options)["server.port"], 65535);
        }
        for value in [json!(65536), json!("65536"), json!(-1), json!("1.0")] {
            assert!(!schema_valid(node, &value, &schema["$defs"]));
            let before = EngineOptions::default();
            let mut options = before.clone();
            assert!(idx
                .merge(&mut options, &json!({"server": {"port": value}}))
                .is_err());
            assert_eq!(options, before);
        }
    }

    /// The generated ECMA surface uses Rust's exact outer trim set; Rust
    /// regex alone is not an independent ECMA validator. The companion JS
    /// check validates actual exported patterns in both RegExp modes.
    #[test]
    fn common_typed_padding_projects_the_runtime_trim_set() {
        fn patterns<'a>(v: &'a Value, out: &mut Vec<&'a str>) {
            if let Some(p) = v.get("pattern").and_then(Value::as_str) {
                out.push(p);
            }
            for name in ["anyOf", "allOf", "oneOf"] {
                if let Some(values) = v.get(name).and_then(Value::as_array) {
                    for value in values {
                        patterns(value, out);
                    }
                }
            }
        }
        let schema = schema();
        let idx = index();
        for (path, value) in [
            ("context.prefix.reuse", "TRUE"),
            ("context.prefix.reuse", "none"),
            ("diagnostics.verbose", "ON"),
            ("memory.admission.drain", "OFF"),
            ("determinism", "FAST"),
            ("generation.top_k", "2"),
            ("generation.temperature", "0.5"),
            ("generation.max_tokens", "16"),
        ] {
            let mut node = &schema;
            for part in path.split('.') {
                node = &node["properties"][part];
            }
            let mut found = Vec::new();
            patterns(node, &mut found);
            assert!(!found.is_empty(), "{path} has no projected string grammar");
            for pattern in found {
                assert!(
                    pattern.starts_with(&format!("^{TRIM_SPACE}*")),
                    "{path}: {pattern}"
                );
                assert!(
                    !pattern.contains(r"\s*"),
                    "{path}: ECMA whitespace differs from Rust trim"
                );
            }
            for (padding, accepted) in [(" \t\n", true), ("\u{0085}", true), ("\u{feff}", false)] {
                let mut options = EngineOptions::default();
                let before = options.clone();
                let input = format!("{padding}{value}{padding}");
                assert_eq!(
                    idx.set(&mut options, path, &input).is_ok(),
                    accepted,
                    "{path}: {input:?}"
                );
                if !accepted {
                    assert_eq!(options, before, "{path}: refused padding changed state");
                }
            }
        }
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
