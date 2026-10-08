//! Option groups: the runtime half of `#[derive(Options)]` (`lloom-macros`).
//!
//! A run selects its mechanisms through option GROUPS — plain structs whose
//! fields are the options (`DecodeConfig`, `Tuning`, …). The derive generates
//! every front-end from the field list (flags, dotted keys, JSON, env
//! aliases, the printed table); this module holds what those front-ends
//! share: the static description of an option ([`KnobMeta`]), the layering
//! helpers, and the value conversions between an option's stored spelling
//! and its key's.
//!
//! Each leaf has two names. Its STORAGE path is its field path through the
//! nested groups (`tuning.qmv.q6k_nr0`), which is where the library crates
//! read it; its KEY is the dotted path in the one option tree a user sees
//! (`kernels.quant_matvec.q6_k.rows_per_simdgroup`), declared on the field
//! (`#[opt(path = "…")]`). Every older spelling — the storage path, the flag,
//! the env variable — stays an alias of the key.
//!
//! Precedence, lowest to highest (each layer writes only what it names):
//! library defaults < the checkpoint < env (`from_env`: the legacy names,
//! then `LLOOM_<KEY>`) < `--config <file>` < `--<group>-config '{json}'` <
//! `--set key=value` < the flags (`Args::apply`). Nothing here runs per
//! token: a group is resolved once, at load or at the request, into the
//! library types the loop already reads.

use serde_json::Value;

/// What an option IS, independent of whether it has a flag: `stable` (user
/// facing, shipping), `advanced` (a shipping tunable with a sensible
/// default), `arm` (an A/B arm whose supersession is undecided), `reference`
/// (a correctness or reference arm, never the fast path), `instrument` (a
/// trace, check, probe or fault injection).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Status {
    Stable,
    Advanced,
    Arm,
    Reference,
    Instrument,
}

impl Status {
    pub const ALL: [Status; 5] = [
        Status::Stable,
        Status::Advanced,
        Status::Arm,
        Status::Reference,
        Status::Instrument,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Status::Stable => "stable",
            Status::Advanced => "advanced",
            Status::Arm => "arm",
            Status::Reference => "reference",
            Status::Instrument => "instrument",
        }
    }

    pub fn parse(s: &str) -> Option<Status> {
        Status::ALL.into_iter().find(|st| st.name() == s)
    }
}

/// A conversion from a key's spelling into a field's stored one.
pub type FromKey = fn(&str) -> Result<String, String>;

/// One option, as the derive describes it. `key` is the field's segment of
/// the storage path and `nested` the sub-group's table for a group-valued
/// field; the rest describes the leaf under its key (`path`).
#[derive(Debug, Clone, Copy)]
pub struct KnobMeta {
    /// The field's storage segment (`q6k_nr0`).
    pub key: &'static str,
    /// The long flag the field has always had (an alias once `cli` is set).
    pub flag: &'static str,
    /// The env variable the field has always read (an alias of `env_key`).
    pub env: Option<&'static str>,
    /// The user-facing description: no section, issue or file references.
    pub doc: &'static str,
    pub variants: &'static [&'static str],
    /// The Rust type, for the table (`usize`, `Option<f64>`, …).
    pub ty: &'static str,
    pub nested: Option<&'static [KnobMeta]>,
    pub is_bool: bool,
    /// The key in the option tree; empty = the storage path is the key.
    pub path: &'static str,
    pub status: Status,
    /// The kind of control (`wait`, `qos`, `lead`, …).
    pub facets: &'static [&'static str],
    /// The data movements it acts on (`token-out`, `weights-in`, …).
    pub flows: &'static [&'static str],
    /// The dedicated long flag, shown in `--help`; empty = reached through
    /// `--set`, `--config`, the env and its (hidden) legacy flag.
    pub cli: &'static str,
    pub short: Option<char>,
    /// Further accepted flag spellings.
    pub aliases: &'static [&'static str],
    /// Valueless flags writing a fixed value (`--cpu` = `backend.device=cpu`).
    pub switches: &'static [(&'static str, &'static str)],
    /// The key is the negation of the stored bool (`memory.admission.drain`
    /// over `no_admission_drain`).
    pub invert: bool,
    /// `(key spelling, stored spelling)` pairs where the two vocabularies
    /// differ (`event` = `true`).
    pub values: &'static [(&'static str, &'static str)],
    /// A key spelling into the stored spelling; stored spellings pass through.
    pub from_v1: Option<FromKey>,
    /// A stored spelling into the key's.
    pub to_v1: Option<fn(&str) -> String>,
    /// The shared value type the key takes (`Wait`, `ThreadQos`, `Locality`,
    /// `Duration`), empty for its own type.
    pub schema: &'static str,
    /// The forms of the shared type this site implements.
    pub forms: &'static [&'static str],
    /// The original doc, with its references (`lloom options --internal`).
    pub internal: &'static str,
    /// The field's spellings are aliases of `path`, but a write to the key
    /// does not target it: it resets to its default instead
    /// (`nodes_per_cb`, `nodes:<k>` of `cut_policy` by another name).
    pub legacy_only: bool,
    /// The storage path is an accepted spelling of the key.
    pub v0: bool,
    /// Further accepted dotted keys, absolute (`model.kv-append`).
    pub legacy_keys: &'static [&'static str],
    /// A nested group whose shorthand spelling (`--readback ring`) is a key.
    pub compound: bool,
}

impl KnobMeta {
    /// The fields every literal starts from.
    pub const BASE: KnobMeta = KnobMeta {
        key: "",
        flag: "",
        env: None,
        doc: "",
        variants: &[],
        ty: "",
        nested: None,
        is_bool: false,
        path: "",
        status: Status::Advanced,
        facets: &[],
        flows: &[],
        cli: "",
        short: None,
        aliases: &[],
        switches: &[],
        invert: false,
        values: &[],
        from_v1: None,
        to_v1: None,
        schema: "",
        forms: &[],
        internal: "",
        legacy_only: false,
        v0: true,
        legacy_keys: &[],
        compound: false,
    };
}

/// `LLOOM_` + the key, upper-cased, `__` between levels.
pub fn canonical_env(path: &str) -> String {
    format!("LLOOM_{}", path.to_ascii_uppercase().replace('.', "__"))
}

/// The option tree's groups, in the order a user meets them, each with the
/// heading `--help` prints it under.
pub const GROUPS: &[(&str, &str)] = &[
    ("model", "Model: what to load"),
    ("weights", "Weights: how they are held and moved"),
    ("context", "Context: length, KV store, prefix reuse"),
    ("memory", "Memory: the quota"),
    ("scheduler", "Scheduler: concurrent sequences"),
    ("prefill", "Prefill"),
    ("decode", "Decode: readback, pacing, threads"),
    ("speculative", "Speculative decoding"),
    ("generation", "Generation: sampling defaults"),
    ("reasoning", "Reasoning"),
    ("prompt", "Prompt"),
    ("determinism", "Determinism"),
    ("executor", "Executor"),
    ("kernels", "Kernels: selection and geometry by op"),
    ("backend", "Backend: device, command stream, buffer pool"),
    ("models", "Model families"),
    ("diagnostics", "Diagnostics"),
    ("server", "Server"),
    ("bench", "Bench"),
];

/// The `--help` heading of the group a key belongs to.
pub fn heading(path: &str) -> &'static str {
    let group = path.split('.').next().unwrap_or(path);
    GROUPS
        .iter()
        .find(|(g, _)| *g == group)
        .map(|(_, h)| *h)
        .unwrap_or("Options")
}

// ---- the value grammar --------------------------------------------------------
//
// One grammar per value type, the same for every flag, dotted key, JSON value
// and env name: a site parses through these, so a spelling means one thing
// wherever it is written. Surrounding whitespace is not part of a typed value,
// and an empty typed value is refused rather than read as unset: an unexpanded
// shell variable must not select the default arm silently.

/// The spellings that mean `true`, any ASCII case.
pub const TRUE_SPELLINGS: &[&str] = &["true", "t", "yes", "y", "1", "on"];
/// The spellings that mean `false`, any ASCII case.
pub const FALSE_SPELLINGS: &[&str] = &["false", "f", "no", "n", "0", "off"];

/// A bool: one of [`TRUE_SPELLINGS`] or [`FALSE_SPELLINGS`], any case.
/// Anything else, the empty value included, is refused.
pub fn parse_bool(value: &str) -> Result<bool, String> {
    let v = value.trim();
    if TRUE_SPELLINGS.iter().any(|s| s.eq_ignore_ascii_case(v)) {
        Ok(true)
    } else if FALSE_SPELLINGS.iter().any(|s| s.eq_ignore_ascii_case(v)) {
        Ok(false)
    } else {
        Err(format!(
            "`{value}` is not a bool ({} or {})",
            TRUE_SPELLINGS.join("|"),
            FALSE_SPELLINGS.join("|")
        ))
    }
}

/// `none`, any case: the spelling that unsets an optional key (JSON's
/// `null` is the other).
pub fn is_none(value: &str) -> bool {
    value.trim().eq_ignore_ascii_case("none")
}

/// The value without its surrounding whitespace; the empty value refused.
pub fn non_empty(value: &str) -> Result<&str, String> {
    let v = value.trim();
    if v.is_empty() {
        Err("an empty value is refused (`none` unsets an optional key)".into())
    } else {
        Ok(v)
    }
}

/// An arm of an enum: one of `arms`, any case; the arm as declared.
pub fn parse_arm<'a>(value: &str, arms: &[&'a str]) -> Result<&'a str, String> {
    let v = non_empty(value)?;
    arms.iter()
        .find(|a| a.eq_ignore_ascii_case(v))
        .copied()
        .ok_or_else(|| format!("`{value}` is not one of {}", arms.join(", ")))
}

/// A count: an unsigned decimal integer.
pub fn parse_count<T: std::str::FromStr>(value: &str) -> Result<T, String> {
    let v = non_empty(value)?;
    v.parse::<T>()
        .map_err(|_| format!("`{value}` is not a count (an unsigned integer)"))
}

/// A number: decimal (`0.7`, `1e-3`) and finite.
pub fn parse_number(value: &str) -> Result<f64, String> {
    let v = non_empty(value)?;
    match v.parse::<f64>() {
        Ok(f) if f.is_finite() => Ok(f),
        _ => Err(format!("`{value}` is not a finite number")),
    }
}

// ---- the meanings that moved ------------------------------------------------------

/// What a legacy env variable's value selected before the variable joined the
/// option tree, where that differs from the grammar: the field's stored value
/// it meant (as JSON text), or `None` where the old reader refused it.
pub type Before = fn(&str) -> Option<String>;

/// A legacy env name, the field it writes (a storage path in its group) and
/// what its values selected before.
pub type BeforeRow = (&'static str, &'static str, Before);

/// The old readers, by shape.
pub mod before {
    /// Read for presence: any value, `0` included, was on.
    pub fn presence(_: &str) -> Option<String> {
        Some("true".into())
    }
    /// Exactly `1` was on, anything else off.
    pub fn exactly_one(v: &str) -> Option<String> {
        Some((v == "1").to_string())
    }
    /// Exactly `true|t|yes|y|1` were on, anything else off.
    pub fn truthy(v: &str) -> Option<String> {
        Some(["true", "t", "yes", "y", "1"].contains(&v).to_string())
    }
    /// A bool field's spellings before the grammar (`1|true|yes|on`,
    /// `0|false|no|off`, any case), anything else refused.
    pub fn derive_bool(v: &str) -> Option<String> {
        match v.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Some("true".into()),
            "0" | "false" | "no" | "off" => Some("false".into()),
            _ => None,
        }
    }
    /// A count parsed as written (no whitespace), anything else refused.
    pub fn strict_count(v: &str) -> Option<String> {
        v.parse::<usize>().ok().map(|n| n.to_string())
    }
    /// A count parsed as written, anything else silently unset.
    pub fn count_or_unset(v: &str) -> Option<String> {
        Some(
            v.parse::<u64>()
                .map(|n| n.to_string())
                .unwrap_or_else(|_| "null".into()),
        )
    }
    /// A path taken as written, the empty one included.
    pub fn path_verbatim(v: &str) -> Option<String> {
        Some(serde_json::Value::String(v.into()).to_string())
    }
    /// An optional typed field: the empty value unset it.
    pub fn empty_unset(v: &str) -> Option<String> {
        v.is_empty().then(|| "null".into())
    }
    /// An optional arm: the empty value and `off` unset it.
    pub fn empty_or_off_unset(v: &str) -> Option<String> {
        (v.is_empty() || v.eq_ignore_ascii_case("off")).then(|| "null".into())
    }
}

/// The declaration at `storage` (a dotted path in the group `metas` describes).
pub fn meta_at(metas: &'static [KnobMeta], storage: &str) -> Option<&'static KnobMeta> {
    let (head, rest) = match storage.split_once('.') {
        Some((h, r)) => (h, Some(r)),
        None => (storage, None),
    };
    let m = metas.iter().find(|m| m.key == head)?;
    match (m.nested, rest) {
        (Some(sub), Some(rest)) => meta_at(sub, rest),
        (None, None) => Some(m),
        _ => None,
    }
}

/// An env lookup by name.
pub type Env<'a> = &'a dyn Fn(&str) -> Option<String>;

/// One line per row of `table` whose variable `env` sets to a value whose
/// meaning moved: the variable, what it selected before, what it selects now.
/// `resolve` builds the group from one variable alone (the env it is given)
/// and returns it as JSON, or the refusal. A cold path: read once, at startup.
pub fn legacy_notes(
    table: &[BeforeRow],
    metas: &'static [KnobMeta],
    env: Env<'_>,
    resolve: &dyn Fn(Env<'_>) -> Result<Value, String>,
) -> Vec<String> {
    let mut out = Vec::new();
    for (var, storage, before) in table {
        let Some(value) = env(var) else { continue };
        let Some(old) = before(&value) else { continue };
        let alone = |name: &str| (name == *var).then(|| value.clone());
        let new = match resolve(&alone) {
            Ok(v) => storage
                .split('.')
                .try_fold(&v, |v, seg| v.get(seg))
                .cloned()
                .unwrap_or(Value::Null)
                .to_string(),
            Err(e) => format!("nothing: it is refused ({e})"),
        };
        if new != old {
            let key = meta_at(metas, storage)
                .map(|m| if m.path.is_empty() { *storage } else { m.path })
                .unwrap_or(storage);
            out.push(format!(
                "[options] {var}={value:?} selected {key} = {old} before the option tree; it now selects {new}"
            ));
        }
    }
    out
}

/// One row of an option table: the full dotted storage path and the leaf's description,
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

/// `Serialize`/`Deserialize` for a knob type through its `Display` and
/// `FromStr` — the flag spelling is the JSON spelling, one parser for both.
#[macro_export]
macro_rules! serde_via_str {
    ($ty:ty) => {
        impl ::serde::Serialize for $ty {
            fn serialize<S: ::serde::Serializer>(&self, s: S) -> ::std::result::Result<S::Ok, S::Error> {
                s.serialize_str(&self.to_string())
            }
        }
        impl<'de> ::serde::Deserialize<'de> for $ty {
            fn deserialize<D: ::serde::Deserializer<'de>>(d: D) -> ::std::result::Result<Self, D::Error> {
                let s = <::std::string::String as ::serde::Deserialize>::deserialize(d)?;
                <$ty as ::std::str::FromStr>::from_str(&s).map_err(::serde::de::Error::custom)
            }
        }
    };
}

/// The conversions between a key's spelling and a field's stored one, for
/// the keys whose vocabulary moved to a shared type (`Wait`, `ThreadQos`).
/// Each `*_from` takes either vocabulary and returns the stored spelling (a
/// stored spelling passes through, for its field's parser to check); each
/// `*_to` renders a stored spelling in the key's. An arm is matched in any
/// case, and a duration's unit too, as every enum and duration is (the value
/// grammar above).
pub mod v1 {

    /// A `Duration` (`50us`, `1ms`, `2s`, the unit any case) in microseconds.
    pub fn duration_us(s: &str) -> Result<u64, String> {
        let lower = s.trim().to_ascii_lowercase();
        let s = lower.as_str();
        let (digits, scale) = if let Some(d) = s.strip_suffix("us") {
            (d, 1)
        } else if let Some(d) = s.strip_suffix("ms") {
            (d, 1_000)
        } else if let Some(d) = s.strip_suffix('s') {
            (d, 1_000_000)
        } else {
            return Err(format!(
                "`{s}` is not a duration (`<n>us`, `<n>ms` or `<n>s`)"
            ));
        };
        let n: u64 = digits
            .parse()
            .map_err(|_| format!("`{s}` is not a duration (`<n>us`, `<n>ms` or `<n>s`)"))?;
        n.checked_mul(scale)
            .ok_or_else(|| format!("`{s}` overflows"))
    }

    /// A `Duration` in whole milliseconds.
    pub fn duration_ms(s: &str) -> Result<u64, String> {
        let us = duration_us(s)?;
        if us % 1_000 != 0 {
            return Err(format!("`{s}` is not a whole number of milliseconds"));
        }
        Ok(us / 1_000)
    }

    fn is_count(s: &str) -> bool {
        !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())
    }

    /// `arm` or `arm:<dur>`, the duration in µs; a bare count is the stored
    /// spelling and passes through.
    fn with_us(arm: &str, rest: Option<&str>) -> Result<String, String> {
        match rest {
            None => Ok(arm.to_string()),
            Some(d) if is_count(d) => Ok(format!("{arm}:{d}")),
            Some(d) => Ok(format!("{arm}:{}", duration_us(d)?)),
        }
    }

    fn split(s: &str) -> (&str, Option<&str>) {
        match s.split_once(':') {
            Some((a, d)) => (a, Some(d)),
            None => (s, None),
        }
    }

    /// `decode.readback.wait`: `sleep:<dur>` is stored `sleep:<µs>`.
    pub fn poll_from(s: &str) -> Result<String, String> {
        let lower = s.trim().to_ascii_lowercase();
        let s = lower.as_str();
        match split(s) {
            ("sleep", rest) => with_us("sleep", rest),
            _ => Ok(s.to_string()),
        }
    }

    pub fn poll_to(s: &str) -> String {
        match split(s) {
            ("sleep", Some(us)) if is_count(us) => format!("sleep:{us}us"),
            _ => s.to_string(),
        }
    }

    /// `decode.host_feed.wait`: `adaptive` is stored `on-demand`, durations
    /// in µs.
    pub fn feed_wait_from(s: &str) -> Result<String, String> {
        let lower = s.trim().to_ascii_lowercase();
        let s = lower.as_str();
        match split(s) {
            ("adaptive", rest) | ("on-demand", rest) => with_us("on-demand", rest),
            ("event", rest) => with_us("event", rest),
            ("sleep", rest) => with_us("sleep", rest),
            _ => Ok(s.to_string()),
        }
    }

    pub fn feed_wait_to(s: &str) -> String {
        match split(s) {
            ("on-demand", None) => "adaptive".to_string(),
            ("on-demand", Some(us)) if is_count(us) => format!("adaptive:{us}us"),
            (arm @ ("event" | "sleep"), Some(us)) if is_count(us) => format!("{arm}:{us}us"),
            _ => s.to_string(),
        }
    }

    /// `backend.metal.completion_wait`: `block` is `synchronize`, `event` the
    /// shared event, `event:<dur>` its timed wait (whole ms), `event-sleep`
    /// and `event-listener` their shared-event arms. `bare_wait_ms` is the
    /// timed wait's default, which a bare `shared-event-wait` renders as.
    pub fn completion_wait_from(s: &str) -> Result<String, String> {
        let lower = s.trim().to_ascii_lowercase();
        let s = lower.as_str();
        let ms = |arm: &str, d: &str| -> Result<String, String> {
            if is_count(d) {
                Ok(format!("{arm}:{d}"))
            } else {
                Ok(format!("{arm}:{}", duration_ms(d)?))
            }
        };
        match split(s) {
            ("block", None) => Ok("synchronize".into()),
            ("event", None) => Ok("shared-event".into()),
            ("event", Some(d)) => ms("shared-event-wait", d),
            ("event-sleep", rest) => with_us("shared-event-sleep", rest),
            ("event-listener", None) => Ok("shared-event-listener".into()),
            ("event-listener", Some(d)) => ms("shared-event-listener", d),
            _ => Ok(s.to_string()),
        }
    }

    pub fn completion_wait_to(s: &str, bare_wait_ms: u64) -> String {
        match split(s) {
            ("synchronize", None) => "block".into(),
            ("shared-event", None) => "event".into(),
            ("shared-event-wait", None) => format!("event:{bare_wait_ms}ms"),
            ("shared-event-wait", Some(ms)) if is_count(ms) => format!("event:{ms}ms"),
            ("shared-event-sleep", None) => "event-sleep".into(),
            ("shared-event-sleep", Some(us)) if is_count(us) => format!("event-sleep:{us}us"),
            ("shared-event-listener", None) => "event-listener".into(),
            ("shared-event-listener", Some(ms)) if is_count(ms) => format!("event-listener:{ms}ms"),
            _ => s.to_string(),
        }
    }

    /// `weights.experts.fetch.wait`: `sleep:<dur>` is stored as the µs count.
    pub fn fetch_wait_from(s: &str) -> Result<String, String> {
        let lower = s.trim().to_ascii_lowercase();
        let s = lower.as_str();
        match split(s) {
            ("sleep", None) => Ok("50".into()),
            ("sleep", Some(d)) if is_count(d) => Ok(d.to_string()),
            ("sleep", Some(d)) => Ok(duration_us(d)?.to_string()),
            _ => Ok(s.to_string()),
        }
    }

    pub fn fetch_wait_to(s: &str) -> String {
        if is_count(s) {
            format!("sleep:{s}us")
        } else {
            s.to_string()
        }
    }

    /// `decode.thread_qos`: `user-interactive` is stored `interactive`.
    pub fn hot_thread_from(s: &str) -> Result<String, String> {
        let lower = s.trim().to_ascii_lowercase();
        let s = lower.as_str();
        Ok(match s {
            "user-interactive" => "interactive".into(),
            other => other.to_string(),
        })
    }

    pub fn hot_thread_to(s: &str) -> String {
        match s {
            "interactive" => "user-interactive".into(),
            "demote" => "default".into(),
            "0" | "none" => "off".into(),
            other => other.to_string(),
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn durations_carry_their_unit() {
            assert_eq!(duration_us("50us"), Ok(50));
            assert_eq!(duration_us("2ms"), Ok(2_000));
            assert_eq!(duration_us("1s"), Ok(1_000_000));
            assert!(duration_us("50").is_err(), "a bare count has no unit");
            assert!(duration_ms("500us").is_err());
            assert_eq!(duration_ms("2s"), Ok(2_000));
        }

        #[test]
        fn each_wait_round_trips_through_its_stored_spelling() {
            for (key, stored) in [
                ("sleep:50us", "sleep:50"),
                ("sleep:1ms", "sleep:1000"),
                ("wfe", "wfe"),
            ] {
                assert_eq!(poll_from(key).unwrap(), stored);
            }
            assert_eq!(poll_to("sleep:50"), "sleep:50us");
            assert_eq!(
                poll_from("sleep:7").unwrap(),
                "sleep:7",
                "the stored spelling passes"
            );
            for (key, stored) in [
                ("adaptive:50us", "on-demand:50"),
                ("adaptive", "on-demand"),
                ("event:1ms", "event:1000"),
                ("spin-hint", "spin-hint"),
            ] {
                assert_eq!(feed_wait_from(key).unwrap(), stored, "{key}");
                assert_eq!(
                    feed_wait_to(stored),
                    key.replace("1ms", "1000us"),
                    "{stored}"
                );
            }
            for (key, stored) in [
                ("block", "synchronize"),
                ("event", "shared-event"),
                ("event:5ms", "shared-event-wait:5"),
                ("event-sleep:50us", "shared-event-sleep:50"),
                ("event-listener:5ms", "shared-event-listener:5"),
            ] {
                assert_eq!(completion_wait_from(key).unwrap(), stored, "{key}");
                assert_eq!(completion_wait_to(stored, 2_000), key, "{stored}");
            }
            assert_eq!(
                completion_wait_to("shared-event-wait", 2_000),
                "event:2000ms"
            );
            assert!(completion_wait_from("event:500us").is_err());
            assert_eq!(fetch_wait_from("sleep:2ms").unwrap(), "2000");
            assert_eq!(fetch_wait_to("50"), "sleep:50us");
            assert_eq!(hot_thread_from("user-interactive").unwrap(), "interactive");
            assert_eq!(hot_thread_to("demote"), "default");
        }
    }
}
