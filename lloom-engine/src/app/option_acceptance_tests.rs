//! Option acceptance through the shipping command and its final layer resolver.
//! No tuning is installed, checkpoint is opened, or device is created here.

use super::{command, resolve_layers, Cli, Command};
use crate::frontend::Frontend;
use crate::option_grammar::tests::{schema_node, schema_valid};
use crate::option_tree::{index, Key};
use crate::EngineOptions;
use clap::FromArgMatches;
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
                let options =
                    result.unwrap_or_else(|e| panic!("{} via {via}: {e}", row["name"]));
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
        .merge(&mut options, &json!({"decode": {"readback": {"wait": null}}}))
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
    assert_eq!(described, ["decode.host_feed.wait", "decode.readback.wait"]);
    assert_eq!(
        index().keys["decode.readback.wait"].meta().grammar,
        Some(&lloom_tensor::options::wait::READBACK_WAIT)
    );
    for path in [
        "backend.metal.completion_wait",
        "weights.experts.fetch.wait",
        "decode.ring.eos_wait",
        "speculative.verify_wait",
    ] {
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

/// These controls pin existing semantics, not approval of the unresolved schemas.
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
        assert!(index().keys[key].meta().grammar.is_none());
        let input = &row["input"];
        let (group, rest) = key.split_once('.').unwrap();
        let flag = format!("--{group}-config");
        let doc = document(rest, input.clone()).to_string();
        let mut routes = vec![("group".into(), config(&[&flag, &doc], &none))];
        if let Some(text) = input.as_str() {
            routes.extend(layers(&index().keys[key], text));
        }
        let accepts = row["accept"] == true
            || (row["accept_without_metal"] == true && !cfg!(feature = "metal"));
        for (via, result) in routes {
            if accepts {
                let options = result.unwrap_or_else(|e| panic!("{} via {via}: {e}", row["name"]));
                let stored = serde_json::to_value(&options).unwrap();
                let actual = row["storage"]
                    .as_str()
                    .unwrap()
                    .split('.')
                    .fold(&stored, |v, s| &v[s]);
                assert_eq!(*actual, row["stored"], "{} via {via}", row["name"]);
                assert_eq!(
                    index().flat(&options)[key],
                    row["rendered"],
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
