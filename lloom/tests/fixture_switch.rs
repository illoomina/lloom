//! The switch reaches the fixture (`lloom::testing::fixture`): this test
//! binary, re-run as a child with `LLOOM_FIXTURE=<value>`, sees the backend
//! the value names — unset or empty is `tiny`, and a value naming none
//! panics rather than run the tiny model under a real-model label.

use lloom::testing::fixture::{Backend, SWITCH};
use std::process::Command;

const CHILD: &str = "LLOOM_FIXTURE_SWITCH_CHILD";

fn child(value: Option<&str>, want: &str) -> bool {
    let mut c = Command::new(std::env::current_exe().expect("the test binary"));
    // The harness names this thread after the test, module path included,
    // so the child filter is right whether this file is its own binary or a
    // module of `tests/main.rs`.
    let me = std::thread::current();
    let name = me.name().expect("a test harness thread").to_string();
    c.args([name.as_str(), "--exact", "--nocapture"])
        .env(CHILD, want);
    match value {
        Some(v) => c.env(SWITCH, v),
        None => c.env_remove(SWITCH),
    };
    // The child's own harness lines stay out of this test's output, where
    // a runner counting `test … FAILED` would read the refused switch's
    // expected failure as this test's.
    c.output().expect("the child ran").status.success()
}

#[test]
fn the_switch_names_the_backend() {
    if let Ok(want) = std::env::var(CHILD) {
        assert_eq!(Backend::current().name(), want);
        return;
    }
    for b in Backend::ALL {
        assert!(
            child(Some(b.name()), b.name()),
            "{SWITCH}={} did not select it",
            b.name()
        );
        assert_eq!(Backend::parse(b.name()), Some(b));
    }
    assert!(child(None, "tiny"), "unset is not tiny");
    assert!(child(Some(""), "tiny"), "empty is not tiny");
    assert!(
        !child(Some("reel"), "tiny"),
        "a mistyped switch ran the tiny model instead of refusing"
    );
    assert!(Backend::Real.at_scale() && Backend::Synthetic.at_scale());
    assert!(!Backend::Tiny.at_scale());
}
