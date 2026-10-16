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
