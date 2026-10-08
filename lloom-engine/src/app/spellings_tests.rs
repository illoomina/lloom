//! Every spelling an option has ever had still reaches it.
//!
//! The reviewed map (`options-v1/rows.v1.json`, 275 rows: every
//! `lloom knobs` row, every flag outside the tree, every env name read
//! directly) names, for each option, its key and the flag, dotted key and
//! env name it had before the tree. Each of those is run through the real
//! front-end — the subcommand's parser and the layer resolution — and must
//! move the row's key and nothing outside it. A spelling that resolves to
//! nothing, or to another option, fails with its row.
//!
//! And every flag the three subcommands printed in their `--help` before
//! the tree (`inputs/*-help.txt`) still parses on the same subcommand.

use super::{resolve_layers, Cli, Command, CommonArgs};
use crate::option_tree::index;
use crate::EngineOptions;
use clap::{CommandFactory, FromArgMatches};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

const ROWS: &str = include_str!("options-v1/rows.v1.json");
const CHAT_HELP: &str = include_str!("options-v1/inputs/chat-help.txt");
const SERVE_HELP: &str = include_str!("options-v1/inputs/serve-help.txt");
const BENCH_HELP: &str = include_str!("options-v1/inputs/bench-help.txt");

const SUBCOMMANDS: [&str; 3] = ["chat", "serve", "bench"];

/// The values a spelling is tried with, until one moves its key.
const POOL: &[&str] = &[
    "true",
    "false",
    "1",
    "2",
    "3",
    "4",
    "5",
    "8",
    "16",
    "32",
    "64",
    "100",
    "256",
    "0",
    "0.5",
    "0.25",
    "1.1",
    "7",
    "x",
    "norm",
    "all",
    "finite",
    "paged",
    "/tmp/lloom-spelling",
    "cpu",
    "resident",
    "io",
    "f16",
    "f32",
    "replay",
    "derived",
    "reorder",
    "host",
    "device",
    "hotlist",
    "inside",
    "now",
    "off",
    "wait",
    "copy",
    "batched",
    "per-row",
    "variant",
    "steps:4",
    "per-token",
    "node-aligned",
    "nodes:4",
    "range",
    "size",
    "plan-wide",
    "steel",
    "half",
    "-O2",
    "bf16",
    "sequential",
    "mtp",
    "dspark",
    "deterministic",
    "checkpoint",
    "truncate",
    "ladder",
    "non-aliasing",
    "always",
    "since-barrier",
    "ring",
    "blocking",
    "host-drain",
    "spin",
    "sleep:7",
    "sleep:100us",
    "event",
    "inherit",
    "adaptive:80us",
    "event:5ms",
    "event-sleep:20us",
    "user-initiated",
    "utility",
    "default",
    "streamed",
    "auto",
    "reuse",
    "concurrent",
    "pipelined",
    "prefill:8",
    "sdpa",
    "flash",
    "int8",
    "in-place",
    "fused",
    "strided",
    "mul",
    "swiglu",
    "reserve",
    "ring:2",
    "rotating:2",
    "crown",
    "pool",
    "low",
    "id:1",
    "3:-5",
    "contiguous",
    "64x64",
    "m",
    "progressive:1,64",
    "inherit:1",
    "recip",
    "hardswish",
];

thread_local! {
    /// The command, built once: a parse clones it rather than rebuilding
    /// every argument.
    static CMD: clap::Command = Cli::command();
}

fn parse<I: IntoIterator<Item = S>, S: Into<std::ffi::OsString> + Clone>(
    argv: I,
) -> Result<Cli, clap::Error> {
    let m = CMD.with(|c| c.clone().try_get_matches_from(argv))?;
    Cli::from_arg_matches(&m)
}

/// Whether `flag` is an argument of `sub` at all.
fn known(sub: &str, flag: &str) -> bool {
    !matches!(
        parse(["lloom", sub, flag]),
        Err(e) if e.kind() == clap::error::ErrorKind::UnknownArgument
    )
}

type Front<'a> = Box<dyn Fn(&mut EngineOptions) -> Result<(), String> + 'a>;

fn front_of(cli: &Cli) -> (&CommonArgs, Front<'_>) {
    match &cli.command {
        Command::Chat(a) => (&a.common, Box::new(move |o| a.front(o))),
        Command::Serve(a) => (&a.common, Box::new(move |o| a.front(o))),
        Command::Bench(a) => (&a.common, Box::new(move |o| a.front(o))),
        Command::Config(a) => (&a.common, Box::new(move |o| a.front.apply(o))),
        Command::Options(_) => unreachable!(),
    }
}

/// The keys `argv` (on a subcommand) and `env` move from their defaults.
fn moved(
    sub: &str,
    argv: &[&str],
    env: &BTreeMap<String, String>,
) -> Result<BTreeSet<String>, String> {
    let mut full = vec!["lloom", sub];
    full.extend_from_slice(argv);
    let cli = parse(full).map_err(|e| e.to_string())?;
    let (common, front) = front_of(&cli);
    let get = |name: &str| env.get(name).cloned();
    let (o, _) = resolve_layers(common, &*front, &get, false).map_err(|e| e.to_string())?;
    let idx = index();
    let before = idx.flat(&EngineOptions::default());
    Ok(idx
        .flat(&o)
        .into_iter()
        .filter(|(k, v)| before.get(k) != Some(v))
        .map(|(k, _)| k)
        .collect())
}

/// The keys a row's key may move with it: the key, its group's keys for a
/// compound shorthand, and the drafter's model beside the method `--mtp`
/// sets.
fn allowed(key: &str) -> impl Fn(&str) -> bool + '_ {
    move |k: &str| {
        k == key
            || k.starts_with(&format!("{key}."))
            || (key == "speculative.method" && k == "speculative.model")
    }
}

/// Try `spell` with each value of the pool; `Ok` with the value that moved
/// the key alone, `Err` with what every try did.
fn reaches(
    key: &str,
    spell: &dyn Fn(&str) -> Result<BTreeSet<String>, String>,
) -> Result<String, String> {
    let ok = allowed(key);
    let mut seen = Vec::new();
    for v in POOL {
        match spell(v) {
            Ok(m) if !m.is_empty() && m.iter().all(|k| ok(k)) => return Ok(v.to_string()),
            Ok(m) if !m.is_empty() => seen.push(format!("{v} moved {m:?}")),
            Ok(_) => {}
            Err(e) => seen.push(format!("{v}: {}", e.lines().next().unwrap_or(""))),
        }
    }
    Err(seen.into_iter().take(6).collect::<Vec<_>>().join(" | "))
}

fn reaches_by_flag(key: &str, flag: &str) -> Result<String, String> {
    let no_env = BTreeMap::new();
    for sub in SUBCOMMANDS.into_iter().filter(|sub| known(sub, flag)) {
        // A valueless spelling first (a switch, a bare bool), then with values.
        if let Ok(m) = moved(sub, &[flag], &no_env) {
            if !m.is_empty() && m.iter().all(|k| allowed(key)(k)) {
                return Ok(format!("{sub} {flag}"));
            }
        }
        let spell = |v: &str| moved(sub, &[flag, v], &no_env);
        if let Ok(v) = reaches(key, &spell) {
            return Ok(format!("{sub} {flag} {v}"));
        }
    }
    Err(format!(
        "{flag}: no subcommand where a value of the pool moves the key alone"
    ))
}

#[test]
fn every_legacy_spelling_in_the_map_reaches_its_key() {
    let rows: Vec<Value> = serde_json::from_str(ROWS).unwrap();
    assert_eq!(rows.len(), 275, "the map's rows");
    let idx = index();
    let no_env = BTreeMap::new();
    let mut failures = Vec::new();
    let mut checked = 0usize;
    for r in &rows {
        let key = r["new"].as_str().unwrap();
        let (old_flag, old_key, old_env) = (
            r["old_flag"].as_str().unwrap_or(""),
            r["old_key"].as_str().unwrap_or(""),
            r["old_env"].as_str().unwrap_or(""),
        );
        if key == "(CLI only)" {
            let parses = SUBCOMMANDS.iter().any(|sub| {
                let full = ["lloom", sub, old_flag, "x"];
                parse(full).is_ok() || parse(&full[..3]).is_ok()
            });
            checked += 1;
            if !parses {
                failures.push(format!("{old_flag}: parses on no subcommand"));
            }
            continue;
        }
        // The heap probe is read only by an ignored test; the env name stays
        // env-only, outside the tree.
        if key == "diagnostics.heap_probe_gb" {
            continue;
        }
        // The SiLU arms are compiled only with Metal.
        if key == "kernels.activation.silu" && !cfg!(feature = "metal") {
            continue;
        }
        if !idx.keys.contains_key(key) {
            failures.push(format!("{key}: not a key of the tree"));
            continue;
        }
        if !old_flag.is_empty() {
            checked += 1;
            if let Err(e) = reaches_by_flag(key, old_flag) {
                failures.push(format!("{key} <- flag {e}"));
            }
        }
        if !old_key.is_empty() {
            checked += 1;
            let spell = |v: &str| {
                let kv = format!("{old_key}={v}");
                moved("chat", &["--set", &kv], &no_env)
            };
            if let Err(e) = reaches(key, &spell) {
                failures.push(format!("{key} <- key {old_key}: {e}"));
            }
        }
        if !old_env.is_empty() {
            checked += 1;
            let spell = |v: &str| {
                let env = BTreeMap::from([(old_env.to_string(), v.to_string())]);
                moved("chat", &[], &env)
            };
            if let Err(e) = reaches(key, &spell) {
                failures.push(format!("{key} <- env {old_env}: {e}"));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {checked} legacy spellings do not reach their key:\n{}",
        failures.len(),
        failures.join("\n")
    );
    eprintln!(
        "[{checked} legacy spellings over {} rows reach their keys]",
        rows.len()
    );
}

/// The long flags a `--help` text lists (`  -t, --temperature <X>`,
/// `      --kind <KIND>`).
fn help_flags(text: &str) -> Vec<String> {
    text.lines()
        .filter_map(|l| {
            let l = l.trim_start();
            let rest = match l.split_once(", --") {
                Some((short, rest)) if short.starts_with('-') && short.len() == 2 => rest,
                _ => l.strip_prefix("--")?,
            };
            let name = rest.split([' ', '[', '<', '=']).next()?;
            (!name.is_empty() && name != "help").then(|| format!("--{name}"))
        })
        .collect()
}

#[test]
fn every_flag_the_help_printed_before_the_tree_still_parses() {
    let mut failures = Vec::new();
    let mut checked = 0usize;
    for (sub, text) in [
        ("chat", CHAT_HELP),
        ("serve", SERVE_HELP),
        ("bench", BENCH_HELP),
    ] {
        let flags = help_flags(text);
        assert!(
            flags.len() > 200,
            "{sub}: {} flags read from its help",
            flags.len()
        );
        for flag in flags {
            // The SiLU arms are compiled only with Metal (the help was a
            // Metal build's).
            if flag == "--silu-impl" && !cfg!(feature = "metal") {
                continue;
            }
            checked += 1;
            let parses = known(sub, &flag)
                && (parse(["lloom", sub, &flag]).is_ok()
                    || POOL.iter().any(|v| parse(["lloom", sub, &flag, v]).is_ok()));
            if !parses {
                failures.push(format!("{sub} {flag}"));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {checked} flags no longer parse: {failures:?}",
        failures.len()
    );
    eprintln!("[{checked} flags of the three subcommands' old help parse]");
}

/// The resolved options of `argv` on `sub`, with no env.
fn resolved(sub: &str, argv: &[&str]) -> Result<EngineOptions, String> {
    let mut full = vec!["lloom", sub];
    full.extend_from_slice(argv);
    let cli = parse(full).map_err(|e| e.to_string())?;
    let (common, front) = front_of(&cli);
    let none = |_: &str| None;
    resolve_layers(common, &*front, &none, false)
        .map(|(o, _)| o)
        .map_err(|e| e.to_string())
}

/// Every flag of a bool key takes the bool grammar the key and the env take
/// (`option_grammar`): the true and false spellings in any case, and the
/// rest refused. The flag is where clap's own bool parser would take
/// only `true|false`.
#[test]
fn every_bool_flag_takes_the_bool_grammar() {
    use lloom_tensor::options::{FALSE_SPELLINGS, TRUE_SPELLINGS};
    const NOT_BOOLS: &[&str] = &["", "2", "tru", "yess", "enable", "none"];
    let idx = index();
    let mut failures = Vec::new();
    let mut checked = 0usize;
    let mut flags_checked = 0usize;
    for key in idx.keys.values().filter(|k| k.compound.is_none()) {
        for leaf in &key.targets {
            let m = leaf.meta;
            if !(m.is_bool || m.ty == "bool" || m.ty == "Option<bool>") {
                continue;
            }
            let mut flags: Vec<&str> = vec![m.flag];
            flags.extend(m.aliases.iter().copied());
            if !m.cli.is_empty() {
                flags.push(m.cli);
            }
            for flag in flags.into_iter().filter(|f| !f.is_empty()) {
                let flag = format!("--{flag}");
                let Some(sub) = SUBCOMMANDS.into_iter().find(|s| known(s, &flag)) else {
                    continue;
                };
                flags_checked += 1;
                let value = |v: &str| -> Result<Option<bool>, String> {
                    let o = resolved(sub, &[&format!("{flag}={v}")])?;
                    let stored = serde_json::to_value(&o).unwrap();
                    let at = leaf
                        .storage
                        .split('.')
                        .try_fold(&stored, |v, seg| v.get(seg))
                        .cloned()
                        .unwrap_or(Value::Null);
                    Ok(match at {
                        Value::Bool(b) => Some(b),
                        Value::String(s) => lloom_tensor::options::parse_bool(&s).ok(),
                        _ => None,
                    })
                };
                for (spellings, want) in [(TRUE_SPELLINGS, true), (FALSE_SPELLINGS, false)] {
                    for s in spellings {
                        for form in [s.to_string(), s.to_ascii_uppercase()] {
                            checked += 1;
                            match value(&form) {
                                Ok(Some(b)) if b == want => {}
                                other => failures.push(format!(
                                    "{} {sub} {flag}={form}: {other:?}, want {want}",
                                    key.path
                                )),
                            }
                        }
                    }
                }
                for s in NOT_BOOLS {
                    checked += 1;
                    if let Ok(v) = value(s) {
                        failures.push(format!("{} {sub} {flag}={s}: accepted ({v:?})", key.path));
                    }
                }
            }
        }
    }
    eprintln!("[{checked} bool flag spellings over {flags_checked} flags]");
    assert!(flags_checked > 80, "{flags_checked} flags");
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
