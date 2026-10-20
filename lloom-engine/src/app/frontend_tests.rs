//! The unset defaults through the real command line: what `chat`, `serve`
//! and `bench` each resolve where no layer named a key; every spelling of
//! such a key naming it — the flag, the dotted keys, the env names, a JSON
//! file and a JSON group — with the library default as good a value as any
//! (the explicit opt-out); the help and the schema saying each front-end's
//! default; and the library tree against the one before the option tree,
//! with each approved exception asserted by number.

use super::{command, resolve_layers, Cli, Command, CommonArgs, Shape};
use crate::frontend::{self, Frontend, When, EXCEPTIONS, ROWS};
use crate::option_tree::index;
use crate::EngineOptions;
use clap::FromArgMatches;
use serde_json::{json, Value};
use std::collections::BTreeMap;

/// The tree `lloom config` printed under an empty environment before the
/// option tree existed.
const BEFORE_TREE: &str = include_str!("options-v1/inputs/config-dba525072.json");
const HELP_BEFORE: [(&str, &str); 3] = [
    ("chat", include_str!("options-v1/inputs/chat-help.txt")),
    ("serve", include_str!("options-v1/inputs/serve-help.txt")),
    ("bench", include_str!("options-v1/inputs/bench-help.txt")),
];

/// Library defaults another change moved on `dev` after `dba525072`, not the
/// option tree: the key, its value before and now, and the change. Recorded
/// so the audit names each; none of them is an exception this tree approved.
const MOVED_ELSEWHERE: &[(&str, &str, &str, &str)] = &[(
    "kernels.quant_matvec.batched_rows",
    "false",
    "true",
    "the row axis, on under `fast`; `deterministic` never takes it (tuning.md §13.5w)",
)];

const PRE_TREE: &str = "the front-end's own value before the option tree";
const TUNED_FLOW: &str = "the tuned decode flow as chat's and serve's default (a08b86f5d, #859)";
const DSPARK_SINGLE: &str = "the maintainer's decision of 2026-10-20 on #1068: the speculative body at one sequence keeps its preset's in-place KV, the paged one gathering the cache every verify round (PR1073 §4e)";
const CHAT_SERVE: &[Frontend] = &[Frontend::Chat, Frontend::Serve];

/// Every row of the front-end table by value: key, front-ends, condition,
/// `before` and where the value comes from. The table is otherwise its own
/// reference, so a row added, removed or moved fails here by name. The rows
/// not `PRE_TREE` were moved by another change on `dev`, recorded so the
/// audit names each; none of them is an exception this tree approved.
const PINNED_ROWS: &[(&str, &[Frontend], When, &str, &str)] = &[
    (
        "context.prefix.reuse",
        &[Frontend::Chat],
        When::Always,
        "true",
        PRE_TREE,
    ),
    (
        "context.prefix.reuse",
        &[Frontend::Serve],
        When::Always,
        "false",
        PRE_TREE,
    ),
    (
        "context.prefix.reuse",
        &[Frontend::Bench],
        When::Always,
        "false",
        PRE_TREE,
    ),
    (
        "generation.max_tokens",
        &[Frontend::Chat],
        When::Always,
        "1024",
        PRE_TREE,
    ),
    (
        "scheduler.prefill_chunk",
        &[Frontend::Serve],
        When::Batched,
        "512",
        PRE_TREE,
    ),
    (
        "scheduler.prefill_chunk",
        &[Frontend::Bench],
        When::Batched,
        "512",
        PRE_TREE,
    ),
    (
        "backend.metal.command_buffer.api",
        CHAT_SERVE,
        When::Metal,
        "metal4",
        TUNED_FLOW,
    ),
    (
        "backend.metal.compiler.merged_unit",
        CHAT_SERVE,
        When::Metal,
        "true",
        TUNED_FLOW,
    ),
    (
        "backend.metal.copy2d_compute",
        CHAT_SERVE,
        When::Metal,
        "true",
        TUNED_FLOW,
    ),
    (
        "backend.metal.heaps.transient",
        CHAT_SERVE,
        When::Metal,
        "true",
        TUNED_FLOW,
    ),
    (
        "decode.readback.kind",
        CHAT_SERVE,
        When::Metal,
        "ring",
        TUNED_FLOW,
    ),
    (
        "decode.readback.eos",
        CHAT_SERVE,
        When::Metal,
        "gpu-batched-nonblocking",
        TUNED_FLOW,
    ),
    (
        "decode.readback.stream_reader",
        CHAT_SERVE,
        When::Metal,
        "true",
        TUNED_FLOW,
    ),
    (
        "models.qwen3_5.eos_gate",
        CHAT_SERVE,
        When::Metal,
        "false",
        PRE_TREE,
    ),
    (
        "models.qwen3_5.eos_gate_cores",
        CHAT_SERVE,
        When::Metal,
        "false",
        PRE_TREE,
    ),
    (
        "decode.readback.queued_steps",
        CHAT_SERVE,
        When::Metal,
        "0",
        PRE_TREE,
    ),
    (
        "kernels.attention.flash.partial_block",
        CHAT_SERVE,
        When::ModelAxes,
        "8",
        TUNED_FLOW,
    ),
    (
        "kernels.fusion.norm_in_gemv",
        CHAT_SERVE,
        When::ModelAxes,
        "true",
        TUNED_FLOW,
    ),
    (
        "models.lfm2.rows_kernels",
        CHAT_SERVE,
        When::ModelAxes,
        "per-row",
        TUNED_FLOW,
    ),
    (
        "speculative.dspark.context_kv",
        CHAT_SERVE,
        When::ModelAxes,
        "paged",
        TUNED_FLOW,
    ),
    (
        "context.kv.layout",
        CHAT_SERVE,
        When::ModelAxes,
        "paged",
        TUNED_FLOW,
    ),
    (
        "context.kv.layout",
        CHAT_SERVE,
        When::DsparkSingle,
        "in-place",
        DSPARK_SINGLE,
    ),
    (
        "kernels.attention.impl",
        CHAT_SERVE,
        When::ModelAxes,
        "flash",
        TUNED_FLOW,
    ),
    (
        "kernels.attention.qkv_layout",
        CHAT_SERVE,
        When::ModelAxes,
        "fused",
        TUNED_FLOW,
    ),
    (
        "kernels.fusion.conv",
        CHAT_SERVE,
        When::ModelAxes,
        "true",
        TUNED_FLOW,
    ),
    (
        "models.lfm2.conv_prefill_context",
        CHAT_SERVE,
        When::ModelAxes,
        "true",
        TUNED_FLOW,
    ),
];

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

/// `lloom <sub> <argv>` with `env`, resolved as `frontend` would.
fn resolve(
    frontend: Option<Frontend>,
    sub: &str,
    argv: &[&str],
    env: &BTreeMap<String, String>,
) -> Result<EngineOptions, String> {
    let mut full = vec!["lloom", sub];
    full.extend_from_slice(argv);
    let m = command()
        .try_get_matches_from(full)
        .map_err(|e| e.to_string())?;
    let cli = Cli::from_arg_matches(&m).map_err(|e| e.to_string())?;
    let (common, front) = front_of(&cli);
    let get = |n: &str| env.get(n).cloned();
    resolve_layers(common, &*front, &get, false, frontend)
        .map(|(o, _)| o)
        .map_err(|e| e.to_string())
}

fn at(o: &EngineOptions, key: &str) -> Value {
    index().flat(o).remove(key).unwrap_or(Value::Null)
}

/// `value` in `key`'s vocabulary, as the tree renders it.
fn typed(key: &str, value: &str) -> Value {
    let mut o = EngineOptions::default();
    index().set(&mut o, key, value).unwrap();
    at(&o, key)
}

/// The library tree before the option tree, read back into today's, and
/// the fields the dump wrote.
fn before_tree() -> (EngineOptions, Vec<String>) {
    #[allow(unused_mut)]
    let mut before: Value = serde_json::from_str(BEFORE_TREE).unwrap();
    // The dump was taken on aarch64, where the poll's default is `wfe`; the
    // grammar refuses `wfe` elsewhere (`unknown poll mechanism`), so off
    // aarch64 the key is left unset and resolves to that platform's own
    // default — the one platform-specific value the dump carries.
    #[cfg(not(target_arch = "aarch64"))]
    if let Some(r) = before
        .pointer_mut("/decode/readback")
        .and_then(Value::as_object_mut)
    {
        r.remove("poll");
    }
    let mut o = EngineOptions::default();
    let written = index().merge_written(&mut o, &before).unwrap();
    (o, written.into_iter().collect())
}

/// The arguments that put a front-end at a batch where a row needs one.
fn batch_for(r: &frontend::Row) -> Vec<&'static str> {
    match r.when {
        When::Batched => vec!["--batch", "4"],
        // No `--model`: the LFM2 snapshot, whose family takes the model axes.
        When::Always | When::Metal | When::ModelAxes => vec![],
        When::DsparkSingle => vec!["--dspark"],
    }
}

/// Each front-end resolves its own default of every key it resolves itself,
/// and the shape it loads with carries it.
#[test]
fn each_front_end_resolves_its_own_defaults_through_the_command_line() {
    let none = BTreeMap::new();
    let mut checked = 0;
    for r in ROWS {
        let argv = batch_for(r);
        let o = resolve(Some(r.frontend), r.frontend.name(), &argv, &none).unwrap();
        if r.when.holds(&o) {
            assert_eq!(
                at(&o, r.key),
                typed(r.key, frontend::value(r)),
                "{} {}",
                r.frontend.name(),
                r.key
            );
        } else {
            // A row whose condition this build cannot meet (the model axes
            // without Metal) leaves the library's own value.
            assert_eq!(
                at(&o, r.key),
                at(&EngineOptions::default(), r.key),
                "{} {} (row not held)",
                r.frontend.name(),
                r.key
            );
        }
        checked += 1;
        if r.key == "context.prefix.reuse" {
            let shape = Shape::of(&o, r.frontend, String::new());
            assert_eq!(
                Value::Bool(shape.reuse_prefix),
                typed(r.key, frontend::value(r))
            );
        }
        // Without its front-end, the library's own value.
        let lib = resolve(None, r.frontend.name(), &argv, &none).unwrap();
        assert_eq!(at(&lib, r.key), at(&EngineOptions::default(), r.key));
    }
    assert_eq!(checked, ROWS.len());
    // `--cpu` keeps the model axes at the library's values on every
    // front-end: the paged pool, FlashDecoding and the fused conv are Metal
    // kernels, and a CPU load must not be refused by its own defaults.
    for f in Frontend::ALL {
        for argv in [&["--cpu"][..], &["--cpu", "--dspark"]] {
            let o = resolve(Some(f), f.name(), argv, &none).unwrap();
            for r in ROWS.iter().filter(|r| {
                r.frontend == f
                    && matches!(r.when, When::Metal | When::ModelAxes | When::DsparkSingle)
            }) {
                assert!(
                    !r.when.holds(&o),
                    "{} {} holds on {argv:?}",
                    f.name(),
                    r.key
                );
                assert_eq!(
                    at(&o, r.key),
                    at(&EngineOptions::default(), r.key),
                    "{} {} on {argv:?}",
                    f.name(),
                    r.key
                );
            }
        }
    }
    // The shape a front-end loads with, from options no front-end resolved
    // (a harness's, a test's): the front-end's own reuse still applies.
    for f in Frontend::ALL {
        let shape = Shape::of(&EngineOptions::default(), f, String::new());
        let want = typed(
            "context.prefix.reuse",
            frontend::value(frontend::row("context.prefix.reuse", f).unwrap()),
        );
        assert_eq!(Value::Bool(shape.reuse_prefix), want, "{}", f.name());
    }
    // Not at a batch, the CHUNK stays the whole prompt on every front-end.
    for f in [Frontend::Serve, Frontend::Bench] {
        let o = resolve(Some(f), f.name(), &[], &none).unwrap();
        assert_eq!(o.decode.prefill.chunk, 0, "{}", f.name());
    }
    // `lloom config --frontend serve` resolves as serve does.
    let m = command()
        .try_get_matches_from(["lloom", "config", "--frontend", "serve"])
        .unwrap();
    let Command::Config(c) = Cli::from_arg_matches(&m).unwrap().command else {
        panic!("config")
    };
    assert_eq!(c.frontend, Some(Frontend::Serve));
}

/// The speculative body at one sequence takes the in-place KV through the
/// command line on chat and serve, after the paged row; the KV given by any
/// spelling holds; at a batch, on the pool preset, without a drafter, with
/// `auto` (resolved at load, after the rows) and on bench, the row does not
/// hold.
#[test]
fn the_speculative_body_at_one_sequence_keeps_the_in_place_kv() {
    let none = BTreeMap::new();
    let key = "context.kv.layout";
    let kv =
        |f: Frontend, argv: &[&str]| at(&resolve(Some(f), f.name(), argv, &none).unwrap(), key);
    let library = at(&EngineOptions::default(), key);
    let metal = When::ModelAxes.holds(&resolve(Some(Frontend::Chat), "chat", &[], &none).unwrap());
    let (in_place, paged) = if metal {
        (typed(key, "in-place"), typed(key, "paged"))
    } else {
        (library.clone(), library.clone())
    };
    for f in [Frontend::Chat, Frontend::Serve] {
        let n = f.name();
        assert_eq!(kv(f, &["--dspark"]), in_place, "{n} --dspark");
        assert_eq!(
            kv(f, &["--speculate", "dspark"]),
            in_place,
            "{n} --speculate dspark"
        );
        assert_eq!(
            kv(f, &["--dspark", "--kv-append", "paged"]),
            typed(key, "paged"),
            "{n} named"
        );
        assert_eq!(
            kv(f, &["--dspark", "--set", "context.kv.layout=cat"]),
            typed(key, "cat"),
            "{n} named by key"
        );
        let env = BTreeMap::from([("LLOOM_CONTEXT__KV__LAYOUT".to_string(), "paged".to_string())]);
        assert_eq!(
            at(&resolve(Some(f), n, &["--dspark"], &env).unwrap(), key),
            typed(key, "paged"),
            "{n} named by env"
        );
        assert_eq!(
            kv(f, &["--dspark", "--set", "scheduler.max_sequences=2"]),
            paged,
            "{n} at a batch"
        );
        assert_eq!(
            kv(f, &["--dspark", "--set", "models.lfm2.preset=rewindable"]),
            in_place,
            "{n} on the rewindable preset"
        );
        assert_eq!(
            kv(f, &["--dspark", "--set", "models.lfm2.preset=pool-spec"]),
            paged,
            "{n} on the pool preset"
        );
        assert_eq!(kv(f, &["--speculate", "auto"]), paged, "{n} auto");
        assert_eq!(kv(f, &[]), paged, "{n} plain");
    }
    assert_eq!(kv(Frontend::Bench, &["--dspark"]), library, "bench");
}

/// The front-end table is the pinned one, row for row; each row another
/// change moved is reported beside the library's value before the option
/// tree, and none of them is an exception.
#[test]
fn every_front_end_row_is_pinned_by_value() {
    let pinned: Vec<(&str, Frontend, When, &str)> = PINNED_ROWS
        .iter()
        .flat_map(|&(k, fs, w, v, _)| fs.iter().map(move |&f| (k, f, w, v)))
        .collect();
    let table: Vec<(&str, Frontend, When, &str)> = ROWS
        .iter()
        .map(|r| (r.key, r.frontend, r.when, r.before))
        .collect();
    let unpinned: Vec<_> = table.iter().filter(|t| !pinned.contains(t)).collect();
    let gone: Vec<_> = pinned.iter().filter(|p| !table.contains(p)).collect();
    assert!(
        unpinned.is_empty() && gone.is_empty(),
        "the front-end table moved; pin each row with where its value comes from\n\
         in the table, not pinned: {unpinned:?}\npinned, not in the table: {gone:?}"
    );
    assert_eq!(table.len(), pinned.len(), "a row appears twice");
    let (before, _) = before_tree();
    for &(k, fs, _, v, why) in PINNED_ROWS.iter().filter(|p| p.4 != PRE_TREE) {
        assert!(
            EXCEPTIONS.iter().all(|e| e.key != k),
            "{k} is an approved exception, not a move recorded here"
        );
        let names: Vec<&str> = fs.iter().map(|f| f.name()).collect();
        eprintln!(
            "[{k}: the library's {} before the option tree; {} now {}: {why}]",
            at(&before, k),
            names.join(", "),
            typed(k, v)
        );
    }
}

/// Every way to name a key a front-end resolves itself is explicit: the
/// value given holds, the library's default and the front-end's own
/// included, through the flags, `--set` by key and by older dotted
/// spelling, `LLOOM_<KEY>` and the older env names, a `--config` file and
/// the group's JSON.
#[test]
fn every_spelling_of_a_front_end_key_names_it_and_the_value_holds() {
    let idx = index();
    let dir = std::env::temp_dir().join(format!("lloom-frontend-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut checked = 0usize;
    let mut bad = Vec::new();
    for r in ROWS {
        let key = idx.keys.get(r.key).unwrap();
        let sub = r.frontend.name();
        let library = at(&EngineOptions::default(), r.key);
        let mut values: Vec<String> = vec![frontend::value(r).to_string()];
        match &library {
            Value::Null => {}
            v => values.push(lloom_tensor::options::render(v)),
        }
        // An enum key (`metal3 | metal4 | auto`) takes another of its arms
        // rather than a number it would refuse.
        let arms: Vec<&str> = key
            .meta()
            .variants
            .iter()
            .copied()
            .filter(|a| !a.contains('<'))
            .collect();
        if key.meta().is_bool || key.meta().ty.contains("bool") {
            values = vec!["true".into(), "false".into()];
        } else if !arms.is_empty() {
            let other = arms
                .iter()
                .find(|a| !values.iter().any(|v| typed(r.key, v) == typed(r.key, a)));
            values.extend(other.map(|a| a.to_string()));
        } else {
            values.push("7".into());
        }
        values.dedup();
        for v in &values {
            let want = typed(r.key, v);
            let mut spellings: Vec<(String, Vec<String>, BTreeMap<String, String>)> = Vec::new();
            let base = batch_for(r)
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>();
            let with = |extra: Vec<String>| {
                let mut a = base.clone();
                a.extend(extra);
                a
            };
            for flag in key.flags() {
                if !key
                    .targets
                    .iter()
                    .flat_map(|l| l.meta.switches.iter())
                    .any(|(sw, _)| format!("--{sw}") == flag)
                {
                    spellings.push((
                        flag.clone(),
                        with(vec![flag.clone(), v.clone()]),
                        BTreeMap::new(),
                    ));
                }
            }
            spellings.push((
                format!("--set {}", r.key),
                with(vec!["--set".into(), format!("{}={v}", r.key)]),
                BTreeMap::new(),
            ));
            for legacy in key.legacy_keys() {
                spellings.push((
                    format!("--set {legacy}"),
                    with(vec!["--set".into(), format!("{legacy}={v}")]),
                    BTreeMap::new(),
                ));
            }
            for env in
                std::iter::once(key.env()).chain(key.legacy_env().iter().map(|e| e.to_string()))
            {
                spellings.push((
                    env.clone(),
                    base.clone(),
                    BTreeMap::from([(env, v.clone())]),
                ));
            }
            let json_value = match &want {
                Value::Null => Value::String(v.clone()),
                w => w.clone(),
            };
            let doc = r
                .key
                .rsplit('.')
                .fold(json_value.clone(), |inner, seg| json!({ seg: inner }));
            let file = dir.join(format!("{}-{sub}-{v}.json", r.key));
            std::fs::write(&file, doc.to_string()).unwrap();
            spellings.push((
                "--config".into(),
                with(vec!["--config".into(), file.display().to_string()]),
                BTreeMap::new(),
            ));
            let (group, rest) = r.key.split_once('.').unwrap();
            let group_doc = rest
                .rsplit('.')
                .fold(json_value, |inner, seg| json!({ seg: inner }));
            spellings.push((
                format!("--{group}-config"),
                with(vec![format!("--{group}-config"), group_doc.to_string()]),
                BTreeMap::new(),
            ));
            for (label, argv, env) in spellings {
                let argv: Vec<&str> = argv.iter().map(String::as_str).collect();
                checked += 1;
                match resolve(Some(r.frontend), sub, &argv, &env) {
                    Ok(o) if at(&o, r.key) == want => {}
                    Ok(o) => bad.push(format!(
                        "{sub} {}: {label} = {v} resolved {}, want {want}",
                        r.key,
                        at(&o, r.key)
                    )),
                    Err(e) => bad.push(format!("{sub} {}: {label} = {v}: {e}", r.key)),
                }
            }
        }
    }
    std::fs::remove_dir_all(&dir).unwrap();
    // The switches: the explicit opt-out and opt-in by their own spellings.
    let none = BTreeMap::new();
    for f in Frontend::ALL {
        let o = resolve(Some(f), f.name(), &["--no-reuse"], &none).unwrap();
        assert_eq!(o.prefix.reuse, Some(false), "{} --no-reuse", f.name());
        let o = resolve(Some(f), f.name(), &["--prefix-reuse"], &none).unwrap();
        assert_eq!(o.prefix.reuse, Some(true), "{} --prefix-reuse", f.name());
        let o = resolve(
            Some(f),
            f.name(),
            &["--set", "context.prefix.reuse=none"],
            &none,
        )
        .unwrap();
        assert_eq!(
            o.prefix.reuse,
            Some(f.reuse_prefix()),
            "{}: `none` is the front-end's own",
            f.name()
        );
    }
    eprintln!("[{checked} spellings of the front-end keys checked]");
    assert!(
        checked >= 90,
        "{checked} spellings checked (93 when written)"
    );
    assert!(
        bad.is_empty(),
        "{} failures:\n{}",
        bad.len(),
        bad.join("\n")
    );
}

/// The unset defaults against the tree before the option tree. The library
/// tree reads back as itself (every value resolved before the tree is today's
/// default); every flag default the subcommands' help printed then is the
/// subcommand's default now; and every difference is an approved exception
/// by number, pending (the old value still resolves) or landed (the new
/// one does). Decisions 43/44 stay opt-in; decisions 32/47's absent, null
/// and explicit values of the session keys stay distinct.
#[test]
fn the_unset_defaults_are_the_trees_before_it_except_the_named_exceptions() {
    let idx = index();
    let none = BTreeMap::new();
    // 1. The library tree.
    let (o, written) = before_tree();
    assert!(written.len() >= 180, "{} fields read", written.len());
    let now = idx.flat(&EngineOptions::default());
    let mut diff: Vec<String> = Vec::new();
    for (k, v) in idx.flat(&o) {
        if now.get(&k) == Some(&v) {
            continue;
        }
        match MOVED_ELSEWHERE.iter().find(|m| m.0 == k) {
            Some(m) if typed(&k, m.1) == v && typed(&k, m.2) == now[&k] => {}
            _ => diff.push(format!("{k}: before {v}, now {}", now[&k])),
        }
    }
    assert!(
        diff.is_empty(),
        "the library defaults moved:\n{}",
        diff.join("\n")
    );
    for m in MOVED_ELSEWHERE {
        assert_eq!(now[m.0], typed(m.0, m.2), "{} is recorded as moved", m.0);
    }

    // 2. Each subcommand's flag defaults, as its help printed them.
    let mut flags_checked = 0;
    let mut moved = Vec::new();
    for (sub, help) in HELP_BEFORE {
        let f = Frontend::parse(sub).unwrap();
        let resolved = resolve(Some(f), sub, &[], &none).unwrap();
        let mut lines = help.lines().peekable();
        while let Some(line) = lines.next() {
            let t = line.trim_start();
            let Some(rest) = t.strip_prefix("--") else {
                continue;
            };
            let flag = format!("--{}", rest.split([' ', '=', '[', '<']).next().unwrap());
            let Some(desc) = lines.peek() else { continue };
            let Some(default) = desc
                .split("[default: ")
                .nth(1)
                .and_then(|d| d.split(']').next())
            else {
                continue;
            };
            let Some(key) = idx.keys.values().find(|k| k.flags().contains(&flag)) else {
                continue;
            };
            let mut probe = resolved.clone();
            idx.set(&mut probe, &key.path, default)
                .unwrap_or_else(|e| panic!("{sub} {flag} [default: {default}]: {e}"));
            flags_checked += 1;
            if at(&probe, &key.path) != at(&resolved, &key.path) {
                let landed = EXCEPTIONS
                    .iter()
                    .any(|e| e.landed && e.key == key.path && e.frontend == f);
                if !landed {
                    moved.push(format!(
                        "{sub} {flag} ({}): [default: {default}] then, {} now",
                        key.path,
                        at(&resolved, &key.path)
                    ));
                }
            }
        }
    }
    assert!(
        flags_checked >= 35,
        "{flags_checked} flag defaults read (38 when written)"
    );
    assert!(
        moved.is_empty(),
        "a flag's default moved:\n{}",
        moved.join("\n")
    );
    eprintln!(
        "[{} library fields and {flags_checked} flag defaults read back]",
        written.len()
    );

    // 3. The approved exceptions, by number.
    for e in EXCEPTIONS {
        let r = frontend::row(e.key, e.frontend).unwrap();
        let o = resolve(Some(e.frontend), e.frontend.name(), &batch_for(r), &none).unwrap();
        // A landed exception resolves to its `after` where its row holds; a
        // row a build cannot hold (the Metal rows on the CPU-only x86 gate)
        // leaves the library's value, which is the row's `before`.
        let want = if e.landed && r.when.holds(&o) { e.after } else { e.before };
        assert_eq!(
            at(&o, e.key),
            typed(e.key, want),
            "decision {} ({}, landed = {})",
            e.decision,
            e.owner,
            e.landed
        );
    }
    let decided: Vec<u32> = EXCEPTIONS.iter().map(|e| e.decision).collect();
    assert_eq!(
        decided,
        [36, 43, 43, 43, 43, 35, 35],
        "the approved default changes: decision 36 (serve reuse), 43 (the gate and its cores arm, chat and serve), 35 (the lead, chat and serve)"
    );
    // 4. Decision 44: the EOS watch stays opt-in on every front-end.
    // Decisions 43/35 (2026-10-19): the gate, its cores arm and the lead of 4
    // are chat's and serve's defaults where the Metal rows hold — a Metal
    // build, not `--cpu` — and the library's values on bench and on the CPU.
    for f in Frontend::ALL {
        let o = resolve(Some(f), f.name(), &[], &none).unwrap();
        assert_eq!(at(&o, "decode.ring.eos_watch"), false, "{}", f.name());
        let on = matches!(f, Frontend::Chat | Frontend::Serve) && When::Metal.holds(&o);
        assert_eq!(at(&o, "models.qwen3_5.eos_gate"), on, "{} gate", f.name());
        assert_eq!(
            at(&o, "models.qwen3_5.eos_gate_cores"),
            on,
            "{} cores",
            f.name()
        );
        assert_eq!(
            at(&o, "decode.readback.queued_steps"),
            typed("decode.readback.queued_steps", if on { "4" } else { "0" }),
            "{} lead",
            f.name()
        );
        let cpu = resolve(Some(f), f.name(), &["--cpu"], &none).unwrap();
        assert_eq!(
            at(&cpu, "models.qwen3_5.eos_gate"),
            false,
            "{} gate on --cpu",
            f.name()
        );
    }

    // 5. Decisions 32/47: the session's absent, null and explicit values.
    for f in Frontend::ALL {
        let sub = f.name();
        let o = resolve(Some(f), sub, &[], &none).unwrap();
        assert_eq!(
            o.session.persist_thinking, None,
            "{sub}: absent stays unset; Front resolves the effective persistence"
        );
        assert_eq!(o.session.missing_reasoning, "as-sent");
        assert_eq!(o.session.keep_logprobs, "none");
        let o = resolve(
            Some(f),
            sub,
            &["--session-config", r#"{"persist_thinking": null}"#],
            &none,
        )
        .unwrap();
        assert_eq!(o.session.persist_thinking, None, "{sub}: null unsets");
        let env = BTreeMap::from([("LLOOM_SESSION__PERSIST_THINKING".to_string(), String::new())]);
        assert!(
            resolve(Some(f), sub, &[], &env).is_err(),
            "{sub}: an empty value is not an unset one"
        );
        if f == Frontend::Bench {
            continue;
        }
        let o = resolve(Some(f), sub, &["--strip-thinking"], &none).unwrap();
        assert_eq!(
            o.session.persist_thinking,
            Some(false),
            "{sub}: an explicit strip"
        );
        let o = resolve(Some(f), sub, &["--persist-thinking"], &none).unwrap();
        assert_eq!(o.session.persist_thinking, Some(true));
        let o = resolve(
            Some(f),
            sub,
            &[
                "--strip-thinking",
                "--session-config",
                r#"{"persist_thinking": null}"#,
            ],
            &none,
        )
        .unwrap();
        assert_eq!(
            o.session.persist_thinking,
            Some(false),
            "{sub}: the flag is the top layer"
        );
    }
}

/// The help each subcommand prints names its own default of the keys it
/// resolves itself, and the pending change by number; the flag of a key
/// whose output part needs a compiled capability says which; the schema
/// carries both.
#[test]
fn the_help_and_the_schema_say_each_front_ends_default() {
    let help = |sub: &str| -> String {
        match command().try_get_matches_from(["lloom", sub, "--help"]) {
            Err(e) if e.kind() == clap::error::ErrorKind::DisplayHelp => e.render().to_string(),
            other => panic!("{sub} --help: {other:?}"),
        }
    };
    let line_of = |text: &str, flag: &str| -> String {
        let mut lines = text.lines();
        while let Some(l) = lines.next() {
            if l.trim_start().starts_with(flag) {
                return format!("{l} {}", lines.next().unwrap_or(""));
            }
        }
        panic!("no {flag} in the help")
    };
    let pending = |f: Frontend, key: &str| {
        EXCEPTIONS
            .iter()
            .find(|e| !e.landed && e.frontend == f && e.key == key)
            .map(|e| format!("; decision {}: {} once landed", e.decision, e.after))
            .unwrap_or_default()
    };
    for f in Frontend::ALL {
        let text = help(f.name());
        let reuse = line_of(&text, "--prefix-reuse");
        let want = format!(
            "[unset here: {}{}]",
            frontend::value(frontend::row("context.prefix.reuse", f).unwrap()),
            pending(f, "context.prefix.reuse")
        );
        assert!(reuse.contains(&want), "{}: {reuse}", f.name());
        let verbose = line_of(&text, "-v, --verbose");
        assert!(
            verbose.contains(
                "[the first token's split and the step-time percentiles: the `profile` build only]"
            ),
            "{verbose}"
        );
    }
    assert!(line_of(&help("chat"), "-n, --max-tokens").contains("[unset here: 1024]"));
    assert!(!line_of(&help("serve"), "-n, --max-tokens").contains("unset here"));
    for sub in ["serve", "bench"] {
        assert!(
            line_of(&help(sub), "--prefill-chunk")
                .contains("[unset here: 512 when scheduler.max_sequences > 1]"),
            "{sub}"
        );
    }

    let schema = crate::option_schema::schema();
    let node = |path: &str| -> &Value {
        let mut n = &schema;
        for seg in path.split('.') {
            n = &n["properties"][seg];
        }
        n
    };
    let reuse = node("context.prefix.reuse")["x-lloom-frontend-defaults"]
        .as_array()
        .unwrap();
    assert_eq!(reuse.len(), 3);
    let serve = reuse.iter().find(|e| e["frontend"] == "serve").unwrap();
    let d36 = EXCEPTIONS.iter().find(|e| e.decision == 36).unwrap();
    assert_eq!(serve["value"], Value::Bool(d36.landed));
    assert_eq!(
        serve["change"],
        json!({"decision": 36, "before": false, "after": true, "landed": d36.landed})
    );
    let chunk = node("scheduler.prefill_chunk")["x-lloom-frontend-defaults"][0].clone();
    assert_eq!(chunk["when"], "scheduler.max_sequences > 1");
    assert_eq!(chunk["value"], 512);
    assert_eq!(
        node("diagnostics.verbose")["x-lloom-needs"],
        json!({"capability": "profile", "part": "the first token's split and the step-time percentiles"})
    );
    assert!(node("decode.in_flight_forwards")
        .get("x-lloom-frontend-defaults")
        .is_none());
}
