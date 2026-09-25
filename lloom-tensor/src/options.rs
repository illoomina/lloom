//! Option groups: the runtime half of `#[derive(Options)]` (`lloom-macros`).
//!
//! A run selects its mechanisms through option GROUPS — plain structs whose
//! fields are the knobs (`DecodeConfig`, `Tuning`, …). The derive generates
//! every front-end from the field list (flags, dotted keys, JSON, env
//! aliases, the printed table); this module holds what those front-ends
//! share: the static description of a knob ([`KnobMeta`]), the layering
//! helpers, and the table renderer `lloom knobs` prints.
//!
//! Precedence, lowest to highest (each layer writes only what it names):
//! library defaults < the checkpoint < env aliases (`from_env`) <
//! `--config <file>` (`merge_json`) < `--<group>-config '{json}'` <
//! `--set group.key=value` (`set`) < the shorthand flags (`Args::apply`).
//! Nothing here runs per token: a group is resolved once, at load or at the
//! request, into the library types the loop already reads.

use serde_json::Value;

/// One knob, as the derive describes it: the dotted key within its group,
/// the flag, the env alias, the doc line, the arms. `nested` is the sub-group's
/// own table for a group-valued field.
#[derive(Debug, Clone, Copy)]
pub struct KnobMeta {
    pub key: &'static str,
    pub flag: &'static str,
    pub env: Option<&'static str>,
    pub doc: &'static str,
    pub variants: &'static [&'static str],
    /// The Rust type, for the table (`usize`, `Option<f64>`, …).
    pub ty: &'static str,
    pub nested: Option<&'static [KnobMeta]>,
    pub is_bool: bool,
}

/// The bool spellings every front-end accepts (`1|true|yes|on` / `0|false|no|off`).
pub fn parse_bool(value: &str) -> Result<bool, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        other => Err(format!("`{other}` is not a bool (1/true/yes/on or 0/false/no/off)")),
    }
}

/// One row of the knobs table: the full dotted key and the leaf's description,
/// with the value the given instance holds (rendered from its JSON).
#[derive(Debug, Clone)]
pub struct KnobRow {
    pub key: String,
    pub flag: String,
    pub env: Option<&'static str>,
    pub ty: &'static str,
    pub variants: &'static [&'static str],
    pub doc: &'static str,
    pub value: String,
}

/// Flatten a group's `META` against its serialized value into rows, one per
/// leaf knob, keys dotted from `prefix` (the group's name).
pub fn rows(prefix: &str, meta: &[KnobMeta], value: &Value) -> Vec<KnobRow> {
    let mut out = Vec::new();
    for m in meta {
        let key = if prefix.is_empty() {
            m.key.to_string()
        } else {
            format!("{prefix}.{}", m.key)
        };
        let v = value.get(m.key).cloned().unwrap_or(Value::Null);
        match m.nested {
            Some(sub) => out.extend(rows(&key, sub, &v)),
            None => out.push(KnobRow {
                key,
                flag: format!("--{}", m.flag),
                env: m.env,
                ty: m.ty,
                variants: m.variants,
                doc: m.doc,
                value: render(&v),
            }),
        }
    }
    out
}

/// A JSON value in its flag spelling: strings unquoted, `null` as `none`.
pub fn render(v: &Value) -> String {
    match v {
        Value::Null => "none".to_string(),
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Every env alias a `META` tree declares, for the drift test that compares
/// them against the `env::var` reads in the source.
pub fn env_aliases(meta: &[KnobMeta]) -> Vec<&'static str> {
    let mut out = Vec::new();
    for m in meta {
        if let Some(sub) = m.nested {
            out.extend(env_aliases(sub));
        } else if let Some(e) = m.env {
            out.push(e);
        }
    }
    out
}

/// Parse `key=value` for `--set`.
pub fn split_assignment(s: &str) -> Result<(&str, &str), String> {
    s.split_once('=')
        .map(|(k, v)| (k.trim(), v.trim()))
        .ok_or_else(|| format!("`{s}`: expected key=value"))
}
