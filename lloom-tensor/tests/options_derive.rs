//! `#[derive(Options)]`: every front-end of one group writes the same fields.
//!
//! A sample group with each field kind (value, bool, `Option`, a custom
//! parser, a nested group, a skipped field) exercised through the flags
//! (`Args::apply`), the dotted setter, the JSON merge, the env aliases and the
//! table — and the precedence between them.

use clap::Parser;
use lloom_macros::Options;
use lloom_tensor::options::{env_aliases, rows};
use serde::{Deserialize, Serialize};

fn parse_list(s: &str) -> Result<Vec<usize>, String> {
    s.split(',')
        .filter(|p| !p.trim().is_empty())
        .map(|p| p.trim().parse::<usize>().map_err(|e| e.to_string()))
        .collect()
}

#[derive(Options, Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(default)]
struct Inner {
    /// The inner arm.
    #[opt(variants = ["a", "b"], env = "LLOOM_TEST_INNER_ARM")]
    arm: String,
    /// A depth.
    #[opt(env = "LLOOM_TEST_INNER_DEPTH")]
    depth: usize,
}

impl Default for Inner {
    fn default() -> Self {
        Inner {
            arm: "a".into(),
            depth: 2,
        }
    }
}

#[derive(Options, Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(default)]
struct Group {
    /// The lookahead.
    #[opt(flag = "look", env = "LLOOM_TEST_LOOK")]
    lookahead: usize,
    /// A switch.
    #[opt(bool)]
    adaptive: bool,
    /// An optional fraction.
    #[opt]
    fraction: Option<f64>,
    /// A schedule.
    #[opt(parser = "parse_list")]
    schedule: Vec<usize>,
    /// The nested group.
    #[opt(nested)]
    inner: Inner,
    /// Not on any front-end.
    #[opt(skip)]
    computed: u32,
    /// Untagged: not on any front-end either.
    untagged: u32,
}

impl Default for Group {
    fn default() -> Self {
        Group {
            lookahead: 1,
            adaptive: false,
            fraction: None,
            schedule: vec![],
            inner: Inner::default(),
            computed: 7,
            untagged: 9,
        }
    }
}

#[derive(Parser, Debug)]
struct Cli {
    #[command(flatten)]
    group: GroupArgs,
}

#[test]
fn flags_write_only_what_was_given() {
    let cli = Cli::parse_from(["t", "--look", "0", "--adaptive", "--arm", "b", "--schedule", "1,16,32"]);
    let mut g = Group::default();
    cli.group.apply(&mut g).unwrap();
    assert_eq!(g.lookahead, 0);
    assert!(g.adaptive);
    assert_eq!(g.fraction, None, "an absent flag keeps the base's value");
    assert_eq!(g.schedule, vec![1, 16, 32]);
    assert_eq!(g.inner.arm, "b");
    assert_eq!(g.inner.depth, 2, "an absent nested flag keeps the base's value");
    assert_eq!(g.computed, 7);

    let cli = Cli::parse_from(["t", "--adaptive=false", "--fraction", "0.5", "--depth", "4"]);
    let mut g = Group {
        adaptive: true,
        ..Group::default()
    };
    cli.group.apply(&mut g).unwrap();
    assert!(!g.adaptive, "`--flag=false` clears a switch");
    assert_eq!(g.fraction, Some(0.5));
    assert_eq!(g.inner.depth, 4);
}

#[test]
fn a_bad_flag_value_names_the_key() {
    let cli = Cli::parse_from(["t", "--look", "many"]);
    let err = cli.group.apply(&mut Group::default()).unwrap_err();
    assert!(err.starts_with("lookahead:"), "{err}");
}

#[test]
fn the_dotted_setter_reaches_every_leaf() {
    let mut g = Group::default();
    g.set("lookahead", "3").unwrap();
    g.set("adaptive", "yes").unwrap();
    g.set("fraction", "0.25").unwrap();
    g.set("fraction", "none").unwrap();
    g.set("schedule", "64").unwrap();
    g.set("inner.arm", "b").unwrap();
    g.set("inner.depth", "8").unwrap();
    assert_eq!(
        g,
        Group {
            lookahead: 3,
            adaptive: true,
            fraction: None,
            schedule: vec![64],
            inner: Inner {
                arm: "b".into(),
                depth: 8
            },
            ..Group::default()
        }
    );
    let err = g.set("inner", "x").unwrap_err();
    assert!(err.contains("is a group"), "{err}");
    let err = g.set("nope", "x").unwrap_err();
    assert!(err.contains("unknown key `nope`") && err.contains("lookahead"), "{err}");
    let err = g.set("computed", "1").unwrap_err();
    assert!(err.contains("unknown key"), "a skipped field is not settable: {err}");
    let err = g.set("lookahead.x", "1").unwrap_err();
    assert!(err.contains("takes no sub-key"), "{err}");
}

#[test]
fn json_merges_field_by_field() {
    let mut g = Group::default();
    g.merge_json(&serde_json::json!({"lookahead": 0, "inner": {"depth": 5}, "fraction": 0.1}))
        .unwrap();
    assert_eq!(g.lookahead, 0);
    assert_eq!(g.inner.depth, 5);
    assert_eq!(g.inner.arm, "a", "a nested key the object does not name is kept");
    assert_eq!(g.fraction, Some(0.1));
    let err = g.merge_json(&serde_json::json!({"nope": 1})).unwrap_err();
    assert!(err.contains("unknown key `nope`"), "{err}");
    let err = g.merge_json(&serde_json::json!(3)).unwrap_err();
    assert!(err.contains("expected a JSON object"), "{err}");
    // The round trip: what `config --print` emits is what `--config` reads.
    let dumped = serde_json::to_value(&g).unwrap();
    let mut back = Group::default();
    back.merge_json(&dumped).unwrap();
    assert_eq!(back, g);
}

#[test]
fn env_aliases_are_a_layer() {
    std::env::set_var("LLOOM_TEST_LOOK", "0");
    std::env::set_var("LLOOM_TEST_INNER_ARM", "b");
    std::env::remove_var("LLOOM_TEST_INNER_DEPTH");
    let mut g = Group::default();
    g.from_env().unwrap();
    assert_eq!(g.lookahead, 0);
    assert_eq!(g.inner.arm, "b");
    assert_eq!(g.inner.depth, 2, "an unset alias keeps the value");
    std::env::set_var("LLOOM_TEST_INNER_DEPTH", "deep");
    let err = Group::default().from_env().unwrap_err();
    assert!(err.starts_with("LLOOM_TEST_INNER_DEPTH:"), "{err}");
    std::env::remove_var("LLOOM_TEST_LOOK");
    std::env::remove_var("LLOOM_TEST_INNER_ARM");
    std::env::remove_var("LLOOM_TEST_INNER_DEPTH");
    assert_eq!(
        env_aliases(Group::META),
        vec!["LLOOM_TEST_LOOK", "LLOOM_TEST_INNER_ARM", "LLOOM_TEST_INNER_DEPTH"]
    );
}

#[test]
fn the_table_lists_every_leaf_with_its_value() {
    let g = Group {
        fraction: Some(0.5),
        ..Group::default()
    };
    let table = rows("group", Group::META, &serde_json::to_value(&g).unwrap());
    let keys: Vec<&str> = table.iter().map(|r| r.key.as_str()).collect();
    assert_eq!(
        keys,
        vec![
            "group.lookahead",
            "group.adaptive",
            "group.fraction",
            "group.schedule",
            "group.inner.arm",
            "group.inner.depth"
        ]
    );
    let look = &table[0];
    assert_eq!(look.flag, "--look");
    assert_eq!(look.env, Some("LLOOM_TEST_LOOK"));
    assert_eq!(look.value, "1");
    assert_eq!(look.doc, "The lookahead.");
    assert_eq!(table[2].value, "0.5");
    assert_eq!(table[4].variants, &["a", "b"]);
    assert_eq!(table[4].ty, "String");
}

#[test]
fn precedence_is_env_then_json_then_set_then_flags() {
    std::env::set_var("LLOOM_TEST_LOOK", "5");
    let mut g = Group::default();
    g.from_env().unwrap();
    assert_eq!(g.lookahead, 5);
    g.merge_json(&serde_json::json!({"lookahead": 4})).unwrap();
    assert_eq!(g.lookahead, 4);
    g.set("lookahead", "3").unwrap();
    assert_eq!(g.lookahead, 3);
    Cli::parse_from(["t", "--look", "2"]).group.apply(&mut g).unwrap();
    assert_eq!(g.lookahead, 2);
    std::env::remove_var("LLOOM_TEST_LOOK");
}
