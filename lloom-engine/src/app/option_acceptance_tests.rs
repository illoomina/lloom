//! Option acceptance through the shipping command and its final layer resolver.
//! No tuning is installed, checkpoint is opened, or device is created here.

use super::{command, resolve_layers, Cli, Command};
use crate::frontend::Frontend;
use crate::option_grammar::tests::{schema_node, schema_valid};
use crate::option_tree::{index, Key};
use crate::EngineOptions;
use clap::FromArgMatches;
use lloom_tensor::utils::metal_is_available;
use serde_json::{json, Value};
use std::collections::BTreeMap;

thread_local! {
    static CMD: clap::Command = command();
}

fn parse(argv: &[&str]) -> Result<Cli, String> {
    let matches = CMD
        .with(|cmd| cmd.clone().try_get_matches_from(argv))
        .map_err(|e| e.to_string())?;
    Cli::from_arg_matches(&matches).map_err(|e| e.to_string())
}

fn config(argv: &[&str], env: &BTreeMap<String, String>) -> Result<EngineOptions, String> {
    let mut full = vec!["lloom", "config"];
    full.extend_from_slice(argv);
    let Command::Config(args) = parse(&full)?.command else {
        unreachable!()
    };
    resolve_layers(
        &args.common,
        &|o| args.front.apply(o),
        &|name| env.get(name).cloned(),
        false,
        args.frontend,
    )
    .map(|(o, _)| o)
    .map_err(|e| e.to_string())
}

fn document(path: &str, value: Value) -> Value {
    path.rsplit('.')
        .fold(value, |inner, segment| json!({segment: inner}))
}

fn stored_bool(o: &EngineOptions, path: &str) -> Option<bool> {
    let stored = serde_json::to_value(o).unwrap();
    let value = path.split('.').try_fold(&stored, |v, s| v.get(s))?;
    match value {
        Value::Bool(b) => Some(*b),
        Value::String(s) => lloom_tensor::options::parse_bool(s).ok(),
        _ => None,
    }
}

/// These are independent layers, not calls to the underlying field parser.
/// The JSON arm also checks the generated value schema with the same validator
/// as the exhaustive leaf-grammar tests.
fn layers(key: &Key, value: &str) -> Vec<(String, Result<EngineOptions, String>)> {
    let none = BTreeMap::new();
    let (group, rest) = key.path.split_once('.').unwrap();
    let group_flag = format!("--{group}-config");
    let group_json = document(rest, json!(value)).to_string();
    vec![
        (
            "--set".into(),
            config(&["--set", &format!("{}={value}", key.path)], &none),
        ),
        (
            key.env(),
            config(&[], &BTreeMap::from([(key.env(), value.to_string())])),
        ),
        (
            group_flag.clone(),
            config(&[&group_flag, &group_json], &none),
        ),
    ]
}

#[test]
fn every_bool_key_keeps_the_grammar_through_final_layer_resolution() {
    const TRUE: &[&str] = &["true", "t", "yes", "y", "1", "on"];
    const FALSE: &[&str] = &["false", "f", "no", "n", "0", "off"];
    let mut direct = 0;
    for axis in lloom::models::lfm2::Config::AXES_META
        .iter()
        .filter(|axis| axis.is_bool)
    {
        for key in [axis.key.to_string(), format!("axes.{}", axis.key)] {
            let mut model = crate::options::ModelConfig::default();
            model.set(&key, "on").unwrap();
            let before = model.clone();
            for value in ["", " \t\r\n "] {
                let error = model.set(&key, value).unwrap_err();
                assert!(error.contains(axis.key), "{key}: {value:?}: {error}");
                assert!(
                    error.contains(&TRUE.join("|")) && error.contains(&FALSE.join("|")),
                    "{key}: {value:?}: {error}"
                );
                assert_eq!(model, before, "{key}: {value:?} changed a refused axis");
                direct += 1;
            }
        }
    }
    assert!(direct > 0, "no bool axes tested directly");
    eprintln!("[{direct} direct ModelConfig::set empty/whitespace refusals]");
    let schema = crate::option_schema::schema();
    let mut keys = 0;
    let mut resolved = 0;
    for key in index().keys.values().filter(|key| key.compound.is_none()) {
        let m = key.meta();
        if !(m.is_bool || m.ty == "bool" || m.ty == "Option<bool>") {
            continue;
        }
        keys += 1;
        let node = schema_node(&schema, &key.path);
        for (spellings, want) in [(TRUE, true), (FALSE, false)] {
            for spelling in spellings {
                let value = format!(" {} ", spelling.to_ascii_uppercase());
                assert!(
                    schema_valid(node, &json!(value), &schema["$defs"]),
                    "{}: {value:?}",
                    key.path
                );
                for (via, result) in layers(key, &value) {
                    let o =
                        result.unwrap_or_else(|e| panic!("{} via {via}: {value:?}: {e}", key.path));
                    for leaf in &key.targets {
                        assert_eq!(
                            stored_bool(&o, &leaf.storage),
                            Some(want ^ leaf.meta.invert),
                            "{} via {via}: {value:?} at {}",
                            key.path,
                            leaf.storage
                        );
                    }
                    resolved += 1;
                }
            }
        }
        for value in ["", "maybe", "2"] {
            assert!(
                !schema_valid(node, &json!(value), &schema["$defs"]),
                "{}: {value:?}",
                key.path
            );
            for (via, result) in layers(key, value) {
                let error = result.unwrap_err();
                assert!(
                    error.contains(m.key)
                        || error.contains(&key.path)
                        || error.contains(&key.env()),
                    "{} via {via}: {error}",
                    key.path
                );
                assert!(
                    error.contains("true") && error.contains("false"),
                    "{} via {via}: {error}",
                    key.path
                );
                resolved += 1;
            }
        }
    }
    assert!(keys > 80, "only {keys} bool keys");
    assert_eq!(resolved, keys * 45);
    eprintln!("[{keys} bool keys; {resolved} final layer resolutions]");
}

#[test]
fn shared_wait_parameters_and_legacy_values_resolve_to_absolute_references() {
    let schema = crate::option_schema::schema();
    let none = BTreeMap::new();
    let rows = [
        (
            "decode.readback.wait",
            " SLEEP:2MS ",
            "sleep:2000",
            "sleep:2000us",
        ),
        (
            "decode.host_feed.wait",
            " ADAPTIVE:2MS ",
            "on-demand:2000",
            "adaptive:2000us",
        ),
        (
            "backend.metal.completion_wait",
            " EVENT:2MS ",
            "shared-event-wait:2",
            "event:2ms",
        ),
        (
            "backend.metal.completion_wait",
            " EVENT-SLEEP:2MS ",
            "shared-event-sleep:2000",
            "event-sleep:2000us",
        ),
        (
            "backend.metal.completion_wait",
            " EVENT-LISTENER:2S ",
            "shared-event-listener:2000",
            "event-listener:2000ms",
        ),
        (
            "weights.experts.fetch.wait",
            " SLEEP:2MS ",
            "2000",
            "sleep:2000us",
        ),
        (
            "decode.thread_qos",
            " USER-INTERACTIVE ",
            "interactive",
            "user-interactive",
        ),
    ];
    let mut checked = 0;
    for (path, value, legacy, want) in rows {
        let key = &index().keys[path];
        assert!(
            schema_valid(schema_node(&schema, path), &json!(value), &schema["$defs"]),
            "{path}: {value:?}"
        );
        let expected = config(&["--set", &format!("{path}={value}")], &none).unwrap();
        assert_eq!(index().flat(&expected)[path], json!(want), "{path}");
        for (via, result) in layers(key, value)
            .into_iter()
            .chain(key.legacy_keys().iter().map(|old| {
                (
                    old.clone(),
                    config(&["--set", &format!("{old}={legacy}")], &none),
                )
            }))
            .chain(key.legacy_env().iter().map(|old| {
                (
                    old.to_string(),
                    config(
                        &[],
                        &BTreeMap::from([(old.to_string(), legacy.to_string())]),
                    ),
                )
            }))
            .chain(
                key.flags()
                    .iter()
                    .map(|flag| (flag.clone(), config(&[&format!("{flag}={legacy}")], &none))),
            )
        {
            let o = result.unwrap_or_else(|e| panic!("{path} via {via}: {e}"));
            assert_eq!(o, expected, "{path} via {via}");
            checked += 1;
        }
    }
    assert!(checked >= rows.len() * 4, "only {checked} spellings");
    eprintln!("[{checked} shared value spellings resolved]");
}

/// The three engine commands of the behavior matrix, without running a model.
/// A path is opaque to resolution; this deliberately nonexistent path must not
/// be opened. The other eleven matrix arms belong to the example harnesses.
#[test]
fn the_three_engine_chat_gate_commands_keep_their_resolved_options() {
    const PARIS: &str = "What is the capital of France? Answer in one short sentence.";
    const PROSE: &str = "Write a short story about a lighthouse keeper who finds a message in a bottle. Use plain prose, at least three paragraphs.";
    const MODEL: &str = "/not-loaded/Qwen-IQ3_XXS-00001-of-00002.gguf";
    let render = |q| {
        format!("<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    };
    let paris = render(PARIS);
    let prose = render(PROSE);
    let arms = [
        (
            "chat-lfm2-paris",
            vec![
                "--profile",
                "deterministic",
                "--sampling-defaults",
                "greedy",
                "--raw",
                "--max-tokens",
                "32",
                "-p",
                PARIS,
            ],
            32,
            PARIS,
            false,
        ),
        (
            "chat-paris",
            vec![
                "--model",
                MODEL,
                "--experts",
                "stream",
                "--profile",
                "deterministic",
                "--raw",
                "--max-tokens",
                "32",
                "-p",
                &paris,
            ],
            32,
            paris.as_str(),
            true,
        ),
        (
            "chat-prose",
            vec![
                "--model",
                MODEL,
                "--experts",
                "stream",
                "--profile",
                "deterministic",
                "--raw",
                "--max-tokens",
                "160",
                "-p",
                &prose,
            ],
            160,
            prose.as_str(),
            true,
        ),
    ];
    let dir = std::env::temp_dir().join(format!(
        "lloom-gate-options-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&dir).unwrap();
    let file = dir.join("resolved.json");
    for (name, argv, max_tokens, prompt, qwen) in &arms {
        let mut full = vec!["lloom", "chat"];
        full.extend_from_slice(argv);
        let Command::Chat(args) = parse(&full).unwrap().command else {
            unreachable!()
        };
        assert!(args.raw, "{name}");
        assert_eq!(args.prompt.as_deref(), Some(*prompt), "{name}");
        let (o, _) = resolve_layers(
            &args.common,
            &|o| args.front(o),
            &|_| None,
            false,
            Some(Frontend::Chat),
        )
        .unwrap();
        let flat = index().flat(&o);
        assert_eq!(flat["determinism"], "deterministic", "{name}");
        assert_eq!(flat["generation.max_tokens"], json!(max_tokens), "{name}");
        assert_eq!(
            flat["model.path"],
            if *qwen { json!(MODEL) } else { Value::Null },
            "{name}"
        );
        if *qwen {
            assert_eq!(flat["weights.experts.placement"], "stream", "{name}");
        } else {
            assert_eq!(o.sampling.defaults, "greedy", "{name}");
        }
        std::fs::write(&file, index().render(&o).to_string()).unwrap();
        let back = config(&["--config", file.to_str().unwrap()], &BTreeMap::new()).unwrap();
        assert_eq!(back, o, "{name}: the resolved tree through --config");
    }
    std::fs::remove_file(file).unwrap();
    std::fs::remove_dir(dir).unwrap();
    assert_eq!(arms.len(), 3);
}

/// Compound string spellings through final resolution, independently of schema.
#[test]
fn compound_readback_forms_reach_the_final_resolver() {
    // Independent literals, not outputs of apply_spelling or SPELLINGS.
    let rows = [
        ("auto", "auto", "poll", 1, true),
        ("blocking", "blocking", "poll", 1, true),
        ("ring", "ring", "blocking", 1, true),
        ("ring-nonblocking", "ring", "poll", 1, true),
        ("ring-fixed", "ring", "poll", 1, false),
        ("ring-la0", "ring", "poll", 0, false),
        ("ring-hostdrain", "ring", "host-drain", 1, true),
        ("ring-gpubatch", "ring", "gpu-batched", 1, true),
        (
            "ring-gpubatch-nonblocking",
            "ring",
            "gpu-batched-nonblocking",
            1,
            true,
        ),
    ];
    let none = BTreeMap::new();
    let mut checked = 0;
    for (spelling, kind, eos, lookahead, adaptive) in rows {
        for value in [
            spelling.to_owned(),
            spelling.to_ascii_uppercase(),
            format!(" \t{spelling}\n "),
        ] {
            let doc = json!({"readback": value}).to_string();
            let full = json!({"decode": {"readback": value}});
            // Temp file is per-process and per-test; no model/checkpoint opened.
            let path = std::env::temp_dir().join(format!(
                "lloom-933-compound-runtime-{}.json",
                std::process::id()
            ));
            std::fs::write(&path, full.to_string()).unwrap();
            let routes = [
                config(&["--set", &format!("decode.readback={value}")], &none),
                config(&["--readback", &value], &none),
                config(&["--decode-config", &doc], &none),
                config(&["--config", path.to_str().unwrap()], &none),
            ];
            std::fs::remove_file(&path).unwrap();
            for result in routes {
                assert!(result.is_ok(), "{value:?}: {result:?}");
                let o = result.unwrap();
                let flat = index().flat(&o);
                assert_eq!(flat["decode.readback.kind"], json!(kind));
                assert_eq!(flat["decode.readback.eos"], json!(eos));
                assert_eq!(flat["decode.readback.lookahead_forwards"], json!(lookahead));
                assert_eq!(flat["decode.readback.adaptive_commit"], json!(adaptive));
                checked += 1;
            }
        }
    }
    for value in ["", " \t ", "ring-x"] {
        let doc = json!({"readback": value}).to_string();
        let path = std::env::temp_dir().join(format!(
            "lloom-933-compound-runtime-invalid-{}.json",
            std::process::id()
        ));
        std::fs::write(&path, json!({"decode": {"readback": value}}).to_string()).unwrap();
        let routes = [
            config(&["--set", &format!("decode.readback={value}")], &none),
            config(&["--readback", value], &none),
            config(&["--decode-config", &doc], &none),
            config(&["--config", path.to_str().unwrap()], &none),
        ];
        std::fs::remove_file(path).unwrap();
        for result in routes {
            let error = result.unwrap_err();
            assert!(error.contains("readback"), "{value:?}: {error}");
        }
    }
    // A field flag remains the last writer after the compound preset.
    let o = config(&["--readback", " RING-LA0 ", "--lookahead", "1"], &none).unwrap();
    assert_eq!(o.decode.readback.lookahead, 1);
    assert!(!o.decode.readback.adaptive_commit);
    // Fields outside a preset's assignment set survive a direct production call.
    let mut seed = crate::ReadbackConfig {
        run_ahead: 17,
        wake_cadence: 13,
        ..Default::default()
    };
    let mut expected = seed.clone();
    expected.kind = crate::ReadbackKind::Ring;
    expected.eos = crate::EosKind::Poll;
    expected.lookahead = 1;
    expected.adaptive_commit = false;
    crate::ReadbackConfig::apply_spelling(&mut seed, " RING-FIXED ").unwrap();
    assert_eq!(seed, expected);
    let before = seed.clone();
    for value in ["", " \t ", "ring-x"] {
        let error = crate::ReadbackConfig::apply_spelling(&mut seed, value).unwrap_err();
        assert!(error.contains("readback"), "{value:?}: {error}");
        let reason = if value.trim().is_empty() {
            "an empty value is refused"
        } else {
            "is not one of"
        };
        assert!(error.contains(reason), "{value:?}: {error}");
        assert_eq!(seed, before, "{value:?}: refused spelling changed fields");
    }
    assert_eq!(checked, 108);
    eprintln!("[{checked} compound final resolutions]");
}

/// Pins only the generated shorthand string node, not objects or couplings.
#[test]
fn compound_readback_forms_reach_the_generated_schema() {
    let schema = crate::option_schema::schema();
    let node = schema_node(&schema, "decode.readback");
    let shorthand = node["anyOf"]
        .as_array()
        .unwrap()
        .iter()
        .find(|node| node["type"] == "string" || node.get("anyOf").is_some())
        .unwrap();
    let admitted = [
        "auto",
        "blocking",
        "ring",
        "ring-nonblocking",
        "ring-fixed",
        "ring-la0",
        "ring-hostdrain",
        "ring-gpubatch",
        "ring-gpubatch-nonblocking",
    ];
    let mut checked = 0;
    for spelling in admitted {
        for value in [
            spelling.to_owned(),
            spelling.to_ascii_uppercase(),
            format!(" \t{spelling}\n "),
        ] {
            assert!(
                schema_valid(shorthand, &json!(value), &schema["$defs"]),
                "{value:?}"
            );
            checked += 1;
        }
    }
    for value in ["", " \t ", "ring-x"] {
        assert!(
            !schema_valid(shorthand, &json!(value), &schema["$defs"]),
            "{value:?}"
        );
    }
    assert_eq!(checked, 27);
    eprintln!("[{checked} admitted compound schema strings; 3 refused]");
}

#[test]
fn compound_readback_schema_and_runtime_share_rust_trim() {
    let schema = crate::option_schema::schema();
    let node = &schema_node(&schema, "decode.readback")["anyOf"][1];
    let none = BTreeMap::new();
    for spelling in [
        "auto",
        "blocking",
        "ring",
        "ring-nonblocking",
        "ring-fixed",
        "ring-la0",
        "ring-hostdrain",
        "ring-gpubatch",
        "ring-gpubatch-nonblocking",
    ] {
        for (padding, expected) in [(" \t\n", true), ("\u{0085}", true), ("\u{FEFF}", false)] {
            let value = format!("{padding}{}{padding}", spelling.to_ascii_uppercase());
            let doc = json!({"readback": value}).to_string();
            let actual = config(&["--decode-config", &doc], &none);
            assert_eq!(actual.is_ok(), expected, "runtime {value:?}: {actual:?}");
            assert_eq!(
                schema_valid(node, &json!(value), &schema["$defs"]),
                expected,
                "schema {value:?}"
            );
            if let Ok(actual) = actual {
                let canonical = config(&["--readback", spelling], &none).unwrap();
                assert_eq!(actual, canonical, "{value:?}");
            }
        }
    }
}

fn readback_wait_cases() -> Vec<Value> {
    serde_json::from_str(include_str!("../../tests/fixtures/readback-wait-s1.json")).unwrap()
}

fn wait_case_accepts(row: &Value) -> bool {
    row["accept"] == true && (row["aarch64"] != true || cfg!(target_arch = "aarch64"))
}

fn wait_case_output<'a>(row: &'a Value, kind: &str) -> &'a Value {
    row.get(kind).unwrap_or_else(|| {
        &row[format!(
            "{kind}_{}",
            if cfg!(target_arch = "aarch64") {
                "aarch64"
            } else {
                "other"
            }
        )]
    })
}

/// Literal references pin conversion, storage and rendering separately.
#[test]
fn readback_wait_descriptor_preserves_literal_resolutions() {
    let key = &index().keys["decode.readback.wait"];
    let none = BTreeMap::new();
    let cases = readback_wait_cases();
    assert_eq!(cases.len(), 53);
    for row in &cases {
        let input = &row["input"];
        let doc = json!({"readback": {"wait": input}}).to_string();
        let path = std::env::temp_dir().join(format!(
            "lloom-readback-wait-literals-{}.json",
            std::process::id()
        ));
        std::fs::write(
            &path,
            json!({"decode": {"readback": {"wait": input}}}).to_string(),
        )
        .unwrap();
        let mut routes = vec![
            (
                "group".to_string(),
                config(&["--decode-config", &doc], &none),
            ),
            (
                "file".to_string(),
                config(&["--config", path.to_str().unwrap()], &none),
            ),
        ];
        std::fs::remove_file(path).unwrap();
        if let Some(text) = input.as_str() {
            routes.extend(layers(key, text));
            routes.push(("flag".into(), config(&["--poll", text], &none)));
            routes.push((
                "storage alias".into(),
                config(&["--set", &format!("decode.readback.poll={text}")], &none),
            ));
            if let Some(converted) = row.get("converted") {
                assert_eq!(
                    json!(lloom_tensor::options::v1::poll_from(text).unwrap()),
                    *converted,
                    "{}",
                    row["name"]
                );
            }
        }
        for (via, result) in routes {
            if wait_case_accepts(row) {
                let options = result.unwrap_or_else(|e| panic!("{} via {via}: {e}", row["name"]));
                assert_eq!(
                    serde_json::to_value(&options).unwrap()["decode"]["readback"]["poll"],
                    *wait_case_output(row, "stored"),
                    "{} via {via}",
                    row["name"]
                );
                assert_eq!(
                    index().flat(&options)["decode.readback.wait"],
                    *wait_case_output(row, "rendered"),
                    "{} via {via}",
                    row["name"]
                );
            } else {
                let error = result.unwrap_err();
                assert!(
                    error.contains("readback") || error.contains("poll"),
                    "{} via {via}: {error}",
                    row["name"]
                );
            }
        }
    }
    let mut options = EngineOptions::default();
    index()
        .set(&mut options, "decode.readback.wait", "sleep:7")
        .unwrap();
    let before = options.clone();
    for value in ["", "unknown", "sleep:18446744073709551616", "sleep:+7"] {
        assert!(index()
            .set(&mut options, "decode.readback.wait", value)
            .is_err());
        assert_eq!(options, before, "refused {value:?} changed the field");
    }
    index()
        .merge(
            &mut options,
            &json!({"decode": {"readback": {"wait": null}}}),
        )
        .unwrap();
    assert_eq!(options, EngineOptions::default());
}

/// A limited Rust-helper check, not standard JSON Schema or ECMA validation.
#[test]
fn readback_wait_descriptor_projects_literal_property_language() {
    let schema = crate::option_schema::schema();
    let node = schema_node(&schema, "decode.readback.wait");
    for row in readback_wait_cases() {
        assert_eq!(
            schema_valid(node, &row["input"], &schema["$defs"]),
            wait_case_accepts(&row),
            "{}",
            row["name"]
        );
    }
    let described: Vec<&str> = index()
        .keys
        .values()
        .filter(|key| key.meta().grammar.is_some())
        .map(|key| key.path.as_str())
        .collect();
    assert_eq!(
        described,
        [
            "backend.metal.completion_wait",
            "decode.host_feed.wait",
            "decode.readback.wait",
            "weights.experts.fetch.wait"
        ]
    );
    assert_eq!(
        index().keys["decode.readback.wait"].meta().grammar,
        Some(&lloom_tensor::options::wait::READBACK_WAIT)
    );
    for path in ["decode.ring.eos_wait", "speculative.verify_wait"] {
        assert!(index().keys[path].meta().grammar.is_none(), "{path}");
    }
}

fn host_feed_wait_cases() -> Vec<Value> {
    serde_json::from_str(include_str!("../../tests/fixtures/host-feed-wait-s2.json")).unwrap()
}

/// References come from FeedWait/PollMechanism's stored contract, not the projection.
#[test]
fn host_feed_wait_descriptor_preserves_literal_resolutions() {
    let key = &index().keys["decode.host_feed.wait"];
    let none = BTreeMap::new();
    let cases = host_feed_wait_cases();
    assert_eq!(cases.len(), 95);
    for row in &cases {
        let input = &row["input"];
        let doc = document("host_feed.wait", input.clone()).to_string();
        let path = std::env::temp_dir().join(format!(
            "lloom-host-feed-wait-literals-{}.json",
            std::process::id()
        ));
        std::fs::write(&path, document(&key.path, input.clone()).to_string()).unwrap();
        let mut routes = vec![
            ("group".into(), config(&["--decode-config", &doc], &none)),
            (
                "file".into(),
                config(&["--config", path.to_str().unwrap()], &none),
            ),
        ];
        std::fs::remove_file(path).unwrap();
        if let Some(text) = input.as_str() {
            routes.extend(layers(key, text));
            routes.push(("flag".into(), config(&["--host-feed-poll", text], &none)));
            routes.push((
                "storage alias".into(),
                config(&["--set", &format!("tuning.ring.feed_poll={text}")], &none),
            ));
            routes.push((
                "legacy env".into(),
                config(
                    &[],
                    &BTreeMap::from([("LLOOM_HOST_FEED_POLL".into(), text.into())]),
                ),
            ));
            if let Some(converted) = row.get("converted") {
                assert_eq!(
                    json!(lloom_tensor::options::v1::feed_wait_from(text).unwrap()),
                    *converted,
                    "{}",
                    row["name"]
                );
            }
        }
        for (via, result) in routes {
            if wait_case_accepts(row) {
                let options = result.unwrap_or_else(|e| panic!("{} via {via}: {e}", row["name"]));
                assert_eq!(
                    json!(options.tuning.ring.feed_poll),
                    row["stored"],
                    "{} via {via}",
                    row["name"]
                );
                assert_eq!(
                    index().flat(&options)[&key.path],
                    row["rendered"],
                    "{} via {via}",
                    row["name"]
                );
                // The cold handoff copies this exact stored value to the model crate.
                let tuning = options.tuning.lloom();
                assert_eq!(json!(tuning.ring.feed_poll), row["stored"]);
                assert_eq!(
                    tuning
                        .ring
                        .feed_poll
                        .parse::<lloom::nn::decode::FeedWait>()
                        .unwrap()
                        .to_string(),
                    row["stored"].as_str().unwrap()
                );
            } else {
                let error = result.unwrap_err();
                assert!(
                    error.contains("feed_poll")
                        || error.contains("host_feed")
                        || error.contains("host-feed"),
                    "{} via {via}: {error}",
                    row["name"]
                );
            }
        }
    }
    let mut options = EngineOptions::default();
    index().set(&mut options, &key.path, "event:7us").unwrap();
    let before = options.clone();
    for row in cases.iter().filter(|row| !wait_case_accepts(row)) {
        if let Some(text) = row["input"].as_str() {
            assert!(index().set(&mut options, &key.path, text).is_err());
        }
        assert!(index()
            .merge(&mut options, &document(&key.path, row["input"].clone()))
            .is_err());
        assert_eq!(
            options, before,
            "{}: refused input changed state",
            row["name"]
        );
    }
    index()
        .merge(&mut options, &document(&key.path, Value::Null))
        .unwrap();
    assert_eq!(options, EngineOptions::default());
    assert_eq!(options.tuning.ring.feed_poll, "on-demand:50");
}

/// Limited Rust-helper property checks; standard-validator agreement is UNRUN.
#[test]
fn host_feed_wait_descriptor_projects_literal_property_language() {
    let schema = crate::option_schema::schema();
    let key = &index().keys["decode.host_feed.wait"];
    assert_eq!(
        key.meta().grammar,
        Some(&lloom_tensor::options::wait::HOST_FEED_WAIT)
    );
    let node = schema_node(&schema, &key.path);
    for row in host_feed_wait_cases() {
        assert_eq!(
            schema_valid(node, &row["input"], &schema["$defs"]),
            wait_case_accepts(&row),
            "{}",
            row["name"]
        );
    }
}

/// The frozen S2 reference stays historical; the selected policy has five
/// explicit overrides audited by historical_wait_policy_overrides_are_exact.
#[test]
fn unapplied_wait_contracts_preserve_existing_resolution_boundaries() {
    let rows: Vec<Value> = serde_json::from_str(include_str!(
        "../../tests/fixtures/unapplied-wait-contracts-s2.json"
    ))
    .unwrap();
    assert_eq!(rows.len(), 33);
    let none = BTreeMap::new();
    for row in rows {
        let key = row["key"].as_str().unwrap();
        if key == "backend.metal.completion_wait" {
            assert_eq!(
                index().keys[key].meta().grammar,
                Some(&lloom_tensor::options::wait::COMPLETION_WAIT)
            );
        } else if key == "weights.experts.fetch.wait" {
            assert_eq!(
                index().keys[key].meta().grammar,
                Some(&lloom_tensor::options::wait::FETCH_WAIT)
            );
        } else {
            assert!(index().keys[key].meta().grammar.is_none());
            assert!(index().keys[key].meta().normalize_values);
        }
        let overrides = wait_policy_overrides();
        let selected = overrides.iter().find(|r| r["name"] == row["name"]);
        let policy = selected.unwrap_or(&row);
        let input = &row["input"];
        let (group, rest) = key.split_once('.').unwrap();
        let flag = format!("--{group}-config");
        let doc = document(rest, input.clone()).to_string();
        let mut routes = vec![("group".into(), config(&[&flag, &doc], &none))];
        if let Some(text) = input.as_str() {
            routes.extend(layers(&index().keys[key], text));
        }
        let accepts = policy["accept"] == true
            || (policy["accept_without_metal"] == true && !metal_is_available());
        for (via, result) in routes {
            // --set parses an assignment and trims its value before the leaf
            // setter; JSON/env hand that setter the original string. Literal
            // per-route references preserve both existing contracts.
            let reference = if selected.is_some() {
                policy
            } else if via == "--set" {
                row.get("set_reference").unwrap_or(&row)
            } else {
                &row
            };
            let route_accepts = if reference.get("accept_without_metal").is_some() {
                reference["accept"] == true
                    || (reference["accept_without_metal"] == true && !metal_is_available())
            } else {
                reference["accept"] == true
            };
            if route_accepts {
                let options = result.unwrap_or_else(|e| panic!("{} via {via}: {e}", row["name"]));
                let stored = serde_json::to_value(&options).unwrap();
                let actual = row["storage"]
                    .as_str()
                    .unwrap()
                    .split('.')
                    .fold(&stored, |v, s| &v[s]);
                assert_eq!(*actual, reference["stored"], "{} via {via}", row["name"]);
                assert_eq!(
                    index().flat(&options)[key],
                    reference["rendered"],
                    "{} via {via}",
                    row["name"]
                );
            } else {
                assert!(
                    result.is_err(),
                    "{} via {via}: unexpectedly accepted",
                    row["name"]
                );
            }
        }
        if !accepts {
            let seed = match key {
                "weights.experts.fetch.wait" => "sleep:7us",
                "backend.metal.completion_wait" => "event:3ms",
                _ => "inherit",
            };
            let mut options = EngineOptions::default();
            index().set(&mut options, key, seed).unwrap();
            let before = options.clone();
            assert!(index()
                .merge(&mut options, &document(key, input.clone()))
                .is_err());
            assert_eq!(options, before, "{}: refusal changed state", row["name"]);
        }
    }
}

fn completion_wait_cases() -> Vec<Value> {
    serde_json::from_str(include_str!("../../tests/fixtures/completion-wait-s3.json")).unwrap()
}

fn completion_case_accepts(row: &Value) -> bool {
    row["accept"] == true || (row["accept_without_metal"] == true && !metal_is_available())
}

/// Literal references are independent of the descriptor and generated pattern.
#[test]
fn completion_wait_descriptor_preserves_literal_resolutions() {
    let key = &index().keys["backend.metal.completion_wait"];
    let none = BTreeMap::new();
    let cases = completion_wait_cases();
    assert_eq!(cases.len(), 193);
    for row in &cases {
        let input = &row["input"];
        let doc = document("metal.completion_wait", input.clone()).to_string();
        let path = std::env::temp_dir().join(format!(
            "lloom-completion-wait-literals-{}.json",
            std::process::id()
        ));
        std::fs::write(&path, document(&key.path, input.clone()).to_string()).unwrap();
        let mut routes = vec![
            ("group".into(), config(&["--backend-config", &doc], &none)),
            (
                "file".into(),
                config(&["--config", path.to_str().unwrap()], &none),
            ),
        ];
        std::fs::remove_file(path).unwrap();
        if let Some(text) = input.as_str() {
            routes.extend(layers(key, text));
            routes.push(("flag".into(), config(&["--completion-wait", text], &none)));
            routes.push((
                "storage alias".into(),
                config(
                    &["--set", &format!("tuning.metal.completion_wait={text}")],
                    &none,
                ),
            ));
            routes.push((
                "legacy env".into(),
                config(
                    &[],
                    &BTreeMap::from([("CANDLE_METAL_COMPLETION_WAIT".into(), text.into())]),
                ),
            ));
        }
        for (via, result) in routes {
            if completion_case_accepts(row) {
                let options = result.unwrap_or_else(|e| panic!("{} via {via}: {e}", row["name"]));
                assert_eq!(
                    json!(options.tuning.metal.completion_wait),
                    row["stored"],
                    "{} via {via}",
                    row["name"]
                );
                assert_eq!(
                    index().flat(&options)[&key.path],
                    row["rendered"],
                    "{} via {via}",
                    row["name"]
                );
                let tuning = options.tuning.tensor();
                assert_eq!(json!(tuning.metal.completion_wait), row["stored"]);
                #[cfg(feature = "metal")]
                {
                    let arm =
                        lloom_tensor::kernels::metal::completion::CompletionWait::from_str_arm(
                            &tuning.metal.completion_wait,
                        )
                        .unwrap();
                    assert_eq!(json!(arm.name()), row["arm"], "{}", row["name"]);
                    assert_eq!(json!(arm.period()), row["period"], "{}", row["name"]);
                }
            } else {
                let error = result.unwrap_err();
                assert!(
                    error.contains("completion_wait") || error.contains("completion-wait"),
                    "{} via {via}: {error}",
                    row["name"]
                );
            }
        }
        for spelling in [&key.path[..], "tuning.metal.completion_wait"] {
            let mut options = EngineOptions::default();
            index().set(&mut options, &key.path, "event:3ms").unwrap();
            let before = options.clone();
            let result = index().merge(&mut options, &document(spelling, input.clone()));
            if completion_case_accepts(row) {
                result.unwrap_or_else(|e| panic!("{} direct JSON {spelling}: {e}", row["name"]));
                assert_eq!(json!(options.tuning.metal.completion_wait), row["stored"]);
                assert_eq!(index().flat(&options)[&key.path], row["rendered"]);
            } else {
                let error = result.unwrap_err();
                assert!(
                    error.contains("completion_wait"),
                    "{}: {error}",
                    row["name"]
                );
                assert_eq!(
                    options, before,
                    "{}: refused JSON changed state",
                    row["name"]
                );
                if let Some(text) = input.as_str() {
                    assert!(index().set(&mut options, spelling, text).is_err());
                    assert_eq!(
                        options, before,
                        "{}: refused setter changed state",
                        row["name"]
                    );
                }
            }
        }
    }
    let mut options = EngineOptions::default();
    index().set(&mut options, &key.path, "event:3ms").unwrap();
    index()
        .merge(&mut options, &document(&key.path, Value::Null))
        .unwrap();
    assert_eq!(options, EngineOptions::default());
    assert_eq!(options.tuning.metal.completion_wait, "synchronize");
}

/// The existing Rust helper is not a standard validator or an ECMA engine.
#[test]
fn completion_wait_descriptor_projects_literal_property_language() {
    let schema = crate::option_schema::schema();
    let key = &index().keys["backend.metal.completion_wait"];
    assert_eq!(
        key.meta().grammar,
        Some(&lloom_tensor::options::wait::COMPLETION_WAIT)
    );
    let node = schema_node(&schema, &key.path);
    for row in completion_wait_cases() {
        assert_eq!(
            schema_valid(node, &row["input"], &schema["$defs"]),
            completion_case_accepts(&row),
            "{}",
            row["name"]
        );
    }
    let raw = &node["x-lloom-wait-parameters"][3];
    assert_eq!(raw["spelling"], "shared-event-sleep");
    assert_eq!(raw["raw-maximum"], json!(18446744073709551615u64));
    assert_eq!(raw["raw-plus"], true);
    assert_eq!(raw["raw-trim"], !metal_is_available());
    assert_eq!(node["default"], "block");
}

/// Cargo can enable tensor Metal without enabling the engine's Metal feature.
/// Both parser contracts keep their full assertions on the linked parser's route.
#[test]
fn completion_wait_compiled_parser_keeps_raw_parameter_language() {
    if metal_is_available() {
        completion_wait_metal_keeps_untrimmed_raw_parameter_language();
    } else {
        completion_wait_no_metal_keeps_trimmed_declared_parameter_language();
    }
}

fn completion_wait_metal_keeps_untrimmed_raw_parameter_language() {
    let mut options = EngineOptions::default();
    index()
        .set(
            &mut options,
            "backend.metal.completion_wait",
            "shared-event-wait:+5",
        )
        .unwrap();
    assert_eq!(options.tuning.metal.completion_wait, "shared-event-wait:+5");
    let before = options.clone();
    let error = index()
        .set(
            &mut options,
            "backend.metal.completion_wait",
            "shared-event-wait: +5",
        )
        .unwrap_err();
    assert!(
        error.contains("completion_wait") && error.contains("bad milliseconds"),
        "{error}"
    );
    assert_eq!(options, before);
    let schema = crate::option_schema::schema();
    let node = schema_node(&schema, "backend.metal.completion_wait");
    assert_eq!(
        node["x-lloom-wait-parser"],
        "CompletionWait::from_str_arm (metal)"
    );
    assert!(!schema_valid(
        node,
        &json!("shared-event-wait: +5"),
        &schema["$defs"]
    ));
}

fn completion_wait_no_metal_keeps_trimmed_declared_parameter_language() {
    let mut options = EngineOptions::default();
    index()
        .set(
            &mut options,
            "backend.metal.completion_wait",
            "shared-event-wait: +5",
        )
        .unwrap();
    assert_eq!(
        options.tuning.metal.completion_wait,
        "shared-event-wait: +5"
    );
    assert_eq!(
        index().flat(&options)["backend.metal.completion_wait"],
        "shared-event-wait: +5"
    );
    let before = options.clone();
    let error = index()
        .set(
            &mut options,
            "backend.metal.completion_wait",
            "shared-event-wait:5ms",
        )
        .unwrap_err();
    assert!(
        error.contains("completion_wait") && error.contains("is not one of"),
        "{error}"
    );
    assert_eq!(options, before);
    let schema = crate::option_schema::schema();
    let node = schema_node(&schema, "backend.metal.completion_wait");
    assert_eq!(node["x-lloom-wait-parser"], "declared_arm (no metal)");
    assert!(schema_valid(
        node,
        &json!("shared-event-wait: +5"),
        &schema["$defs"]
    ));
}

#[test]
fn completion_wait_checked_duration_errors_preserve_named_state() {
    let mut options = EngineOptions::default();
    index()
        .set(&mut options, "backend.metal.completion_wait", "event:3ms")
        .unwrap();
    let before = options.clone();
    for (input, reason) in [
        ("event:500us", "not a whole number of milliseconds"),
        (
            "event-listener:1001us",
            "not a whole number of milliseconds",
        ),
        ("event:18446744073709552ms", "overflows"),
        ("event-sleep:18446744073710s", "overflows"),
        ("event:+5", "not a duration"),
    ] {
        let error = index()
            .set(&mut options, "backend.metal.completion_wait", input)
            .unwrap_err();
        assert!(
            error.contains("completion_wait") && error.contains(reason),
            "{input:?}: {error}"
        );
        assert_eq!(options, before, "{input:?}: refused duration changed state");
    }
    for key in [
        "backend.metal.completion_wait.unknown",
        "backend.metal.unknown_wait",
    ] {
        let error = index().set(&mut options, key, "event").unwrap_err();
        assert!(error.contains("unknown key"), "{error}");
        assert_eq!(options, before);
    }
}

fn wait_policy_overrides() -> Vec<Value> {
    serde_json::from_str(include_str!(
        "../../tests/fixtures/wait-normalization-s2-overrides.json"
    ))
    .unwrap()
}

#[test]
fn historical_wait_policy_overrides_are_exact() {
    let old: Vec<Value> = serde_json::from_str(include_str!(
        "../../tests/fixtures/unapplied-wait-contracts-s2.json"
    ))
    .unwrap();
    let new = wait_policy_overrides();
    let names = new
        .iter()
        .map(|r| r["name"].as_str().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(
        names,
        [
            "fetch native integral float",
            "decode event case",
            "decode event padding",
            "speculative event case",
            "speculative event padding"
        ]
    );
    for selected in &new {
        let matches = old
            .iter()
            .filter(|r| r["name"] == selected["name"])
            .collect::<Vec<_>>();
        assert_eq!(matches.len(), 1);
        let prior = matches[0];
        assert_eq!(prior["key"], selected["key"]);
        assert_eq!(prior["input"], selected["input"]);
        assert_eq!(prior["accept"], false);
        assert_eq!(selected["prior_accept"], false);
        assert_eq!(selected["accept"], true);
    }
    assert_eq!(old.len(), 33);
    for name in ["decode event padding", "speculative event padding"] {
        let prior = old.iter().find(|r| r["name"] == name).unwrap();
        assert_eq!(
            prior["set_reference"],
            json!({"accept": true, "stored": true, "rendered": "event"})
        );
    }
}

fn fetch_normalization_cases() -> Vec<Value> {
    serde_json::from_str(include_str!(
        "../../tests/fixtures/fetch-wait-normalization.json"
    ))
    .unwrap()
}

fn fetch_case_accepts(row: &Value) -> bool {
    assert!(
        matches!(usize::BITS, 32 | 64),
        "fixture needs a new target-size reference"
    );
    row[if usize::BITS == 32 {
        "accept32"
    } else {
        "accept64"
    }] == true
}

fn fetch_case_stored(row: &Value) -> u64 {
    row.get("stored")
        .unwrap_or(&row["stored64"])
        .as_str()
        .unwrap()
        .parse()
        .unwrap()
}

fn assert_fetch_result(row: &Value, via: &str, result: Result<EngineOptions, String>) {
    if fetch_case_accepts(row) {
        let options = result.unwrap_or_else(|e| panic!("{} via {via}: {e}", row["name"]));
        let want = fetch_case_stored(row);
        assert_eq!(
            options.tuning.stream.fetch_poll_us as u64, want,
            "{} via {via}",
            row["name"]
        );
        assert_eq!(
            index().flat(&options)["weights.experts.fetch.wait"],
            json!(format!("sleep:{want}us"))
        );
        assert_eq!(options.tuning.lloom().stream.fetch_poll_us as u64, want);
    } else {
        let error = result.unwrap_err();
        assert!(
            error.contains("wait")
                || error.contains("fetch_poll_us")
                || error.contains("fetch-poll-us"),
            "{} via {via}: {error}",
            row["name"]
        );
    }
}

#[test]
fn fetch_wait_native_numbers_and_strings_reach_the_final_resolver() {
    let key = &index().keys["weights.experts.fetch.wait"];
    let rows = fetch_normalization_cases();
    assert_eq!(rows.len(), 58);
    let none = BTreeMap::new();
    for row in &rows {
        let raw = row["input_json"].as_str().unwrap();
        let input: Value = serde_json::from_str(raw).unwrap();
        let group = format!("{{\"experts\":{{\"fetch\":{{\"wait\":{raw}}}}}}}");
        let path = std::env::temp_dir().join(format!(
            "lloom-fetch-normalization-{}.json",
            std::process::id()
        ));
        std::fs::write(&path, format!("{{\"weights\":{group}}}")).unwrap();
        let routes = [
            ("group", config(&["--weights-config", &group], &none)),
            ("file", config(&["--config", path.to_str().unwrap()], &none)),
        ];
        std::fs::remove_file(path).unwrap();
        for (via, result) in routes {
            assert_fetch_result(row, via, result);
        }
        if let Some(text) = input.as_str() {
            for (via, result) in layers(key, text)
                .into_iter()
                .chain(key.legacy_keys().into_iter().map(|old| {
                    (
                        old.clone(),
                        config(&["--set", &format!("{old}={text}")], &none),
                    )
                }))
                .chain(key.legacy_env().into_iter().map(|old| {
                    (
                        old.into(),
                        config(&[], &BTreeMap::from([(old.into(), text.into())])),
                    )
                }))
                .chain(
                    key.flags()
                        .into_iter()
                        .map(|flag| (flag.clone(), config(&[&format!("{flag}={text}")], &none))),
                )
            {
                assert_fetch_result(row, &via, result);
            }
        }
        for spelling in [key.path.as_str(), "tuning.stream.fetch_poll_us"] {
            let mut options = EngineOptions::default();
            index().set(&mut options, &key.path, "sleep:11us").unwrap();
            let before = options.clone();
            let result = index().merge(&mut options, &document(spelling, input.clone()));
            if fetch_case_accepts(row) {
                result.unwrap();
                assert_eq!(
                    options.tuning.stream.fetch_poll_us as u64,
                    fetch_case_stored(row),
                    "{}: {spelling}",
                    row["name"]
                );
            } else {
                assert!(result.is_err(), "{}: {spelling}", row["name"]);
                assert_eq!(
                    options, before,
                    "{}: refused input changed state",
                    row["name"]
                );
            }
        }
    }
}

/// Literal IEEE-754 values, independent of decimal-token rounding in serde.
#[test]
fn fetch_wait_native_precision_neighbors_use_checked_target_bounds() {
    let rows: Vec<Value> = serde_json::from_str(include_str!(
        "../../tests/fixtures/fetch-wait-f64-neighbors.json"
    ))
    .unwrap();
    assert_eq!(rows.len(), 14);
    for row in &rows {
        let bits = u64::from_str_radix(row["bits"].as_str().unwrap(), 16).unwrap();
        let input = Value::Number(serde_json::Number::from_f64(f64::from_bits(bits)).unwrap());
        let mut options = EngineOptions::default();
        index()
            .set(&mut options, "weights.experts.fetch.wait", "sleep:11us")
            .unwrap();
        let before = options.clone();
        let result = index().merge(
            &mut options,
            &document("weights.experts.fetch.wait", input.clone()),
        );
        if fetch_case_accepts(row) {
            result.unwrap();
            assert_eq!(
                options.tuning.stream.fetch_poll_us as u64,
                fetch_case_stored(row),
                "{}",
                row["name"]
            );
        } else {
            assert!(result.is_err(), "{}", row["name"]);
            assert_eq!(options, before, "{}", row["name"]);
        }
        let mut direct = EngineOptions::default();
        let prior = direct.clone();
        let result = direct.merge_json(&document("tuning.stream.fetch_poll_us", input));
        assert_eq!(
            result.is_ok(),
            fetch_case_accepts(row),
            "{}: derive merge",
            row["name"]
        );
        if fetch_case_accepts(row) {
            assert_eq!(
                direct.tuning.stream.fetch_poll_us as u64,
                fetch_case_stored(row)
            );
        } else {
            assert_eq!(direct, prior);
        }
    }
    for f in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(
            serde_json::Number::from_f64(f).is_none(),
            "serde cannot represent a nonfinite JSON number"
        );
    }
    for signed in [-1i64, 0, 1, 7, i64::MAX] {
        let mut options = EngineOptions::default();
        let result = index().merge(
            &mut options,
            &document("weights.experts.fetch.wait", json!(signed)),
        );
        let want = signed > 0 && (usize::BITS == 64 || signed <= 4294967295);
        assert_eq!(result.is_ok(), want, "signed native {signed}");
        if want {
            assert_eq!(options.tuning.stream.fetch_poll_us as u64, signed as u64);
        }
    }
}

#[test]
fn fetch_wait_schema_projects_the_same_positive_usize_domain() {
    let schema = crate::option_schema::schema();
    let key = &index().keys["weights.experts.fetch.wait"];
    assert_eq!(
        key.meta().grammar,
        Some(&lloom_tensor::options::wait::FETCH_WAIT)
    );
    let node = schema_node(&schema, &key.path);
    let native = &node["anyOf"][0]["anyOf"][1];
    assert_eq!(
        *native,
        json!({"type": "integer", "minimum": 1, "maximum": usize::MAX})
    );
    assert_eq!(node["default"], "sleep:50us");
    for row in fetch_normalization_cases() {
        let input: Value = serde_json::from_str(row["input_json"].as_str().unwrap()).unwrap();
        assert_eq!(
            schema_valid(node, &input, &schema["$defs"]),
            fetch_case_accepts(&row),
            "{}: limited property helper",
            row["name"]
        );
    }
}

#[test]
fn fetch_wait_normalization_preserves_text_and_non_target_numeric_contracts() {
    let mut options = EngineOptions::default();
    index()
        .set(&mut options, "weights.experts.fetch.wait", "sleep:11us")
        .unwrap();
    let before = options.clone();
    for text in ["7.0", "7e0", "NaN", "inf", "sleep:+7", "sleep:0us"] {
        let error = index()
            .set(&mut options, "weights.experts.fetch.wait", text)
            .unwrap_err();
        assert!(error.contains("fetch_poll_us"), "{text}: {error}");
        assert_eq!(options, before);
    }
    for path in [
        "weights.experts.fetch.exit_after_steps",
        "tuning.stream.fetch_exit",
    ] {
        let error = index()
            .merge(&mut options, &document(path, json!(7.0)))
            .unwrap_err();
        assert!(error.contains("fetch_exit"), "{error}");
        assert_eq!(options, before, "unrelated count widened");
    }
    for path in [
        "weights.experts.fetch.wait.unknown",
        "weights.experts.unknown_wait",
    ] {
        assert!(index()
            .set(&mut options, path, "7")
            .unwrap_err()
            .contains("unknown key"));
        assert_eq!(options, before);
    }
    let m = index().keys["generation.sampling.method"].meta();
    assert!(!m.normalize_values);
    assert_eq!(
        lloom_tensor::options::map_value("gumbel", m.values, m.normalize_values),
        "true"
    );
    assert_eq!(
        lloom_tensor::options::map_value(" GUMBEL ", m.values, m.normalize_values),
        " GUMBEL "
    );
}

/// Native numeric JSON bools share one ingress normalization; a string's
/// lexical grammar, alias mapping and inverted/storage semantics stay intact.
#[test]
fn native_numeric_bools_share_all_json_ingress_without_widening_text() {
    // The handwritten model group accepts nested, flat and dotted axes.
    fn derive_documents(storage: &str, value: &Value) -> Vec<(String, Value)> {
        let mut documents = vec![(storage.into(), document(storage, value.clone()))];
        if let Some(axis) = storage.strip_prefix("model.axes.") {
            let flat = format!("model.{axis}");
            documents.push((flat.clone(), document(&flat, value.clone())));
            let mut dotted = serde_json::Map::new();
            dotted.insert(format!("axes.{axis}"), value.clone());
            documents.push((
                format!("model[axes.{axis}]"),
                document("model", Value::Object(dotted)),
            ));
        }
        documents
    }
    let accepted = [
        ("0.0", false),
        ("-0.0", false),
        ("0e3", false),
        ("-0e-3", false),
        ("1.0", true),
        ("1e0", true),
        ("10e-1", true),
    ];
    let invalid_text = [
        "-1",
        "2",
        "0.5",
        "1.0000000000000002",
        "5e-324",
        "1e308",
        "\"0.0\"",
        "\"1.0\"",
        "\"1e0\"",
    ];
    let mut invalid: Vec<(String, Value)> = invalid_text
        .into_iter()
        .map(|raw| (raw.into(), serde_json::from_str(raw).unwrap()))
        .collect();
    // Pin represented neighbors independently of the decimal-token parser:
    // without float_roundtrip it can round 0.9999999999999999 to exactly1.
    let one = 1.0_f64;
    for (label, bits, below) in [
        ("represented predecessor of1", one.to_bits() - 1, true),
        ("represented successor of1", one.to_bits() + 1, false),
    ] {
        let number = f64::from_bits(bits);
        assert!(number.is_finite() && number > 0.0 && number != one);
        assert_eq!(number < one, below, "{label}");
        assert_eq!(number > one, !below, "{label}");
        let value = Value::Number(serde_json::Number::from_f64(number).unwrap());
        assert_eq!(value.as_f64().unwrap().to_bits(), bits, "{label}");
        invalid.push((label.into(), value));
    }
    let idx = index();
    let mut keys = 0;
    let mut inverted = 0;
    let mut mapped = 0;
    let mut optional = 0;
    let mut native_writes = 0;
    let mut axis_alias_writes = 0;
    for key in idx.keys.values().filter(|key| key.compound.is_none()) {
        let meta = key.meta();
        if !(meta.is_bool || matches!(meta.ty, "bool" | "Option<bool>")) {
            continue;
        }
        keys += 1;
        inverted += usize::from(key.targets.iter().any(|leaf| leaf.meta.invert));
        mapped += usize::from(!meta.values.is_empty());
        optional += usize::from(
            key.targets
                .iter()
                .any(|leaf| leaf.meta.ty == "Option<bool>"),
        );
        for (raw, want) in accepted {
            let value: Value = serde_json::from_str(raw).unwrap();
            let mut canonical = EngineOptions::default();
            idx.merge(&mut canonical, &document(&key.path, value.clone()))
                .unwrap();
            for leaf in &key.targets {
                assert_eq!(
                    stored_bool(&canonical, &leaf.storage),
                    Some(want ^ leaf.meta.invert),
                    "{} {raw} {}",
                    key.path,
                    leaf.storage
                );
            }
            for spelling in std::iter::once(key.path.clone()).chain(key.legacy_keys()) {
                let mut actual = EngineOptions::default();
                let mut reference = actual.clone();
                idx.merge(&mut actual, &document(&spelling, value.clone()))
                    .unwrap();
                idx.merge(&mut reference, &document(&spelling, json!(want)))
                    .unwrap();
                assert_eq!(actual, reference, "{spelling} native {raw}");
                native_writes += 1;
            }
            for leaf in &key.targets {
                let reference_documents = derive_documents(&leaf.storage, &json!(want));
                for ((spelling, native), (reference_spelling, boolean)) in
                    derive_documents(&leaf.storage, &value)
                        .into_iter()
                        .zip(reference_documents)
                {
                    assert_eq!(spelling, reference_spelling);
                    let mut actual = EngineOptions::default();
                    let mut reference = actual.clone();
                    actual
                        .merge_json(&native)
                        .unwrap_or_else(|error| panic!("{spelling} native {raw}: {error}"));
                    reference.merge_json(&boolean).unwrap();
                    assert_eq!(actual, reference, "{spelling} derive {raw}");
                    native_writes += 1;
                    axis_alias_writes += usize::from(spelling != leaf.storage);
                }
            }
            let (group, rest) = key.path.split_once('.').unwrap();
            let flag = format!("--{group}-config");
            let native = document(rest, value.clone()).to_string();
            let boolean = document(rest, json!(want)).to_string();
            assert_eq!(
                config(&[&flag, &native], &BTreeMap::new()).unwrap(),
                config(&[&flag, &boolean], &BTreeMap::new()).unwrap(),
                "{} final group {raw}",
                key.path
            );
        }
        for spelling in std::iter::once(key.path.clone()).chain(key.legacy_keys()) {
            for (raw, value) in &invalid {
                let mut actual = EngineOptions::default();
                idx.set(&mut actual, "generation.max_tokens", "37").unwrap();
                let prior = actual.clone();
                assert!(
                    idx.merge(&mut actual, &document(&spelling, value.clone()))
                        .is_err(),
                    "{spelling} refused {raw}"
                );
                assert_eq!(actual, prior, "{spelling} refusal changed state");
            }
        }
        for leaf in &key.targets {
            for (raw, value) in &invalid {
                for (spelling, native) in derive_documents(&leaf.storage, value) {
                    let mut actual = EngineOptions::default();
                    let prior = actual.clone();
                    assert!(
                        actual.merge_json(&native).is_err(),
                        "{spelling} derive accepted invalid {raw}"
                    );
                    assert_eq!(actual, prior, "{spelling} derive refusal changed state");
                }
            }
        }
    }
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(serde_json::Number::from_f64(value).is_none());
    }
    assert!(keys > 80 && inverted > 0 && mapped > 0 && optional > 0 && native_writes > 1000 && axis_alias_writes > 0, "keys={keys} inverted={inverted} mapped={mapped} optional={optional} writes={native_writes} axis_alias_writes={axis_alias_writes}");
    eprintln!("NATIVE_BOOL keys={keys} inverted={inverted} mapped={mapped} optional={optional} writes={native_writes} axis_alias_writes={axis_alias_writes}");
}

fn bool_wait_cases() -> Vec<Value> {
    serde_json::from_str(include_str!(
        "../../tests/fixtures/bool-wait-normalization.json"
    ))
    .unwrap()
}

fn bool_wait_config_path(path: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "lloom-bool-wait-{}-{path}.json",
        std::process::id()
    ))
}

#[test]
fn bool_wait_config_files_are_distinct_in_one_process() {
    assert_ne!(
        bool_wait_config_path("decode.ring.eos_wait"),
        bool_wait_config_path("speculative.verify_wait")
    );
}

fn assert_bool_wait_normalization(path: &str, storage: &str, reset: bool) {
    let key = &index().keys[path];
    assert!(key.meta().normalize_values);
    let none = BTreeMap::new();
    let cases = bool_wait_cases();
    assert_eq!(cases.len(), 85);
    for row in &cases {
        let input = &row["input"];
        let want = if row["reset"] == true {
            reset
        } else {
            row["stored"].as_bool().unwrap_or(false)
        };
        let (group, rest) = path.split_once('.').unwrap();
        let flag = format!("--{group}-config");
        let doc = document(rest, input.clone()).to_string();
        let file = bool_wait_config_path(path);
        std::fs::write(&file, document(path, input.clone()).to_string()).unwrap();
        let mut routes = vec![
            ("group".into(), config(&[&flag, &doc], &none)),
            (
                "file".into(),
                config(&["--config", file.to_str().unwrap()], &none),
            ),
        ];
        std::fs::remove_file(file).unwrap();
        if let Some(text) = input.as_str() {
            routes.extend(layers(key, text));
            routes.extend(
                key.flags()
                    .into_iter()
                    .map(|flag| (flag.clone(), config(&[&format!("{flag}={text}")], &none))),
            );
            routes.extend(key.legacy_keys().into_iter().map(|old| {
                (
                    old.clone(),
                    config(&["--set", &format!("{old}={text}")], &none),
                )
            }));
            routes.extend(key.legacy_env().into_iter().map(|old| {
                (
                    old.into(),
                    config(&[], &BTreeMap::from([(old.into(), text.into())])),
                )
            }));
        }
        for (via, result) in routes {
            if row["accept"] == true {
                let options =
                    result.unwrap_or_else(|e| panic!("{path} {} via {via}: {e}", row["name"]));
                assert_eq!(stored_bool(&options, storage), Some(want));
                assert_eq!(
                    index().flat(&options)[path],
                    json!(if want { "event" } else { "inherit" })
                );
                let copied = options.tuning.lloom();
                assert_eq!(
                    match path {
                        "decode.ring.eos_wait" => copied.ring.eos_park,
                        "speculative.verify_wait" => copied.ring.verify_park,
                        _ => panic!("unexpected bool wait field"),
                    },
                    want
                );
            } else {
                assert!(result.is_err(), "{path} {} via {via}", row["name"]);
            }
        }
        for spelling in [path, storage] {
            let mut options = EngineOptions::default();
            index()
                .set(&mut options, path, if reset { "inherit" } else { "event" })
                .unwrap();
            let prior = options.clone();
            let result = index().merge(&mut options, &document(spelling, input.clone()));
            if row["accept"] == true {
                result.unwrap();
                assert_eq!(stored_bool(&options, storage), Some(want));
                if row["reset"] == true {
                    assert_eq!(options, EngineOptions::default());
                }
            } else {
                let error = result.unwrap_err();
                assert!(
                    error.contains(key.meta().key)
                        || error.contains(path)
                        || error.contains("expected a value"),
                    "{error}"
                );
                assert_eq!(options, prior, "{path}: refused input changed state");
            }
        }
    }
    for flag in key.flags() {
        let options = config(&[&flag], &none).unwrap();
        assert_eq!(stored_bool(&options, storage), Some(true), "bare {flag}");
    }
}

#[test]
fn eos_wait_alias_normalization_preserves_false_inherit_reset() {
    assert_bool_wait_normalization("decode.ring.eos_wait", "tuning.ring.eos_park", false);
}

#[test]
fn verify_wait_alias_normalization_preserves_true_event_reset() {
    assert_bool_wait_normalization("speculative.verify_wait", "tuning.ring.verify_park", true);
}

#[test]
fn bool_wait_alias_schema_is_finite_and_normalization_is_opt_in() {
    let schema = crate::option_schema::schema();
    let normalized = index()
        .keys
        .values()
        .filter(|k| k.meta().normalize_values)
        .map(|k| k.path.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        normalized,
        ["decode.ring.eos_wait", "speculative.verify_wait"]
    );
    for (path, default) in [
        ("decode.ring.eos_wait", "inherit"),
        ("speculative.verify_wait", "event"),
    ] {
        let node = schema_node(&schema, path);
        assert_eq!(node["default"], default);
        assert_eq!(node["anyOf"][0]["anyOf"][2], json!({"enum": [0, 1]}));
        for row in bool_wait_cases() {
            if row.get("standard_accept").is_some() {
                // serde_json enum equality in the limited helper is not
                // mathematical JSON Schema equality. Native numeric bools
                // follow standard equality; the independent validator covers it.
                assert_eq!(row["standard_accept"], true);
                assert_eq!(row["accept"], true);
                assert!(row["input"].is_f64());
                assert!(!schema_valid(node, &row["input"], &schema["$defs"]));
            } else {
                assert_eq!(
                    schema_valid(node, &row["input"], &schema["$defs"]),
                    row["accept"] == true,
                    "{path} {}",
                    row["name"]
                );
            }
        }
    }
}

/// Pin the preexisting representation limit without declaring the rounded
/// native number equivalent to its original decimal token.
#[test]
fn fetch_wait_json_number_precision_limit_remains_explicit() {
    let represented: Value = serde_json::from_str("1.00000000000000001").unwrap();
    assert_eq!(represented, json!(1.0));
    let mut options = EngineOptions::default();
    index()
        .merge(
            &mut options,
            &document("weights.experts.fetch.wait", represented),
        )
        .unwrap();
    assert_eq!(options.tuning.stream.fetch_poll_us, 1);
    let before = options.clone();
    assert!(index()
        .merge(
            &mut options,
            &document("weights.experts.fetch.wait", json!("1.00000000000000001"))
        )
        .is_err());
    assert_eq!(options, before);
    let rounded_max: Value = serde_json::from_str("18446744073709551615.0").unwrap();
    assert_eq!(rounded_max, json!(18446744073709551616.0));
    assert!(index()
        .merge(
            &mut options,
            &document("weights.experts.fetch.wait", rounded_max)
        )
        .is_err());
    assert_eq!(options, before);
    for raw in ["NaN", "Infinity", "-Infinity", "1e999"] {
        assert!(
            serde_json::from_str::<Value>(raw).is_err(),
            "not a representable native JSON number: {raw}"
        );
    }
}

#[test]
fn fetch_wait_native_refusals_keep_existing_count_errors_and_state() {
    for input in [json!(0), json!(-1), json!(7.25), json!(0.5), json!(1e100)] {
        let mut options = EngineOptions::default();
        index()
            .set(&mut options, "weights.experts.fetch.wait", "sleep:11us")
            .unwrap();
        let prior = options.clone();
        let text_error = options
            .set("tuning.stream.fetch_poll_us", &input.to_string())
            .unwrap_err();
        let error = index()
            .merge(&mut options, &document("weights.experts.fetch.wait", input))
            .unwrap_err();
        assert_eq!(error, format!("weights.experts.fetch.wait: {text_error}"));
        assert_eq!(options, prior);
    }
}

/// The real serving parser + shared layer resolver, without starting HTTP,
/// opening a checkpoint, installing tuning or creating a device.
fn native_sampler_serving_options(argv: &[&str], env: &BTreeMap<String, String>) -> EngineOptions {
    let mut full = vec!["lloom", "serve"];
    full.extend_from_slice(argv);
    let Command::Serve(args) = parse(&full).expect("actual serve command") .command else {
        unreachable!()
    };
    resolve_layers(&args.common, &|o| args.front(o), &|name| env.get(name).cloned(), false, Some(Frontend::Serve))
        .expect("actual serving layers").0
}

#[test]
fn device_filter_serving_ingress_and_schema_share_one_opt_in() {
    let path = "decode.readback.device_filter";
    let key = index().keys.get(path).expect("generated registry entry");
    assert!(key.flags().contains(&"--device-filter".to_string()));
    let schema = crate::option_schema::schema();
    let node = schema_node(&schema, path);
    for value in [json!(false), json!(true)] {
        assert!(schema_valid(node, &value, &schema["$defs"]));
    }
    assert!(!schema_valid(node, &json!("maybe"), &schema["$defs"]));
    let none = BTreeMap::new();
    assert!(!native_sampler_serving_options(&[], &none).decode.readback.device_filter);
    for argv in [
        vec!["--device-filter"],
        vec!["--set", "decode.readback.device_filter=true"],
        vec!["--decode-config", r#"{"readback":{"device_filter":true}}"#],
    ] {
        assert!(native_sampler_serving_options(&argv, &none).decode.readback.device_filter);
    }
    assert!(native_sampler_serving_options(&[], &BTreeMap::from([(key.env(), "true".into())]))
        .decode.readback.device_filter);
    assert!(!native_sampler_serving_options(&["--device-filter=false"],
        &BTreeMap::from([(key.env(), "true".into())])).decode.readback.device_filter);
}

#[test]
fn device_filter_reports_shared_selection_not_unconditional_host_demotion() {
    let o = native_sampler_serving_options(
        &["--device-filter", "--readback", "ring", "--temperature", "1", "--top-k", "2"],
        &BTreeMap::new(),
    );
    let sampling = o.sampling.sampling().unwrap();
    let params = crate::DecodeParams { readback: o.decode.readback.arm(), ..Default::default() };
    // Unsupported backend uses the exact request resolver; options parsing is
    // not itself model/device capability admission.
    assert!(params.sampler_route(true, &sampling, false, o.decode.readback.ring_arm())
        .err().unwrap().to_string().contains("requires a Metal build and device"));
    let selected = params.sampler_route(true, &sampling, true, o.decode.readback.ring_arm()).unwrap();
    assert!(selected.prepared_ring);
    assert!(matches!(selected.arm, crate::ReadbackArm::Ring(_)));
    let lines = crate::couplings::explain(&o);
    assert!(!lines.iter().any(|line| line.ends_with("(C1)") || line.ends_with("(C2)")));
    if cfg!(feature = "metal") {
        assert!(lines.iter().any(|line| line.contains(&selected.arm.name()) && line.contains("model capability/admission still required")));
    } else {
        assert!(lines.iter().any(|line| line.contains("default request would refuse")));
    }
    let mut host = o.clone();
    host.decode.readback.device_filter = false;
    assert!(crate::couplings::explain(&host).iter().any(|line| line.ends_with("(C2)")));
}
