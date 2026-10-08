//! Process-global state, claimed by the test that installs it.
//!
//! The tuning groups are `OnceLock`s: the first installer wins and a later
//! one is told so, quietly. Two tests sharing a process — the integration
//! suite is one binary (`tests/main.rs`), and plain `cargo test` runs its
//! tests as threads of one process — would run the second under the first's
//! group and pass. The CI runner is `cargo nextest run`, one process per
//! test, where this never fires; this is the loud failure for every other
//! runner (`design/CONTRIBUTING.md` §3.1d).
//!
//! A claim is keyed by the claiming thread's name: the test harness names a
//! test's thread after the test, so two tests are two names. A thread with
//! no name (a pool worker) or the main thread (a binary, or a harness run
//! with `--test-threads=1`) claims nothing, so the engine's own install at
//! start-up and every production read are untouched.

use parking_lot::Mutex;

static OWNER: Mutex<Option<(&'static str, String)>> = Mutex::new(None);

/// The calling test's name, when the thread is a test harness thread.
fn test_name() -> Option<String> {
    let t = std::thread::current();
    let name = t.name()?;
    if name == "main" || name.starts_with("tokio-runtime") {
        return None;
    }
    Some(name.to_string())
}

/// Record that the calling test now owns `what` for the process; a second
/// test claiming anything in the same process is a shared-state hazard and
/// panics by name, naming the runner that avoids it.
pub fn claim(what: &'static str) {
    let Some(me) = test_name() else { return };
    let mut owner = OWNER.lock();
    match owner.as_ref() {
        None => *owner = Some((what, me)),
        Some((theirs, first)) if *first != me => panic!(
            "process-global state: `{what}` is being installed by test `{me}`, but this \
             process's `{theirs}` is owned by test `{first}`. Two tests share a process; run \
             one process per test (`cargo nextest run`, CONTRIBUTING.md §3.1d)"
        ),
        Some(_) => {}
    }
}

/// The test that owns this process's global state, if a claim was made.
pub fn owner() -> Option<String> {
    OWNER.lock().as_ref().map(|(_, name)| name.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A claim from a named non-main thread owns the process; a second name
    /// is refused by name. Run on spawned threads so the harness's own thread
    /// name does not take part.
    #[test]
    fn a_second_test_name_is_refused() {
        let first = std::thread::Builder::new()
            .name("first::test".into())
            .spawn(|| claim("tuning"))
            .unwrap()
            .join();
        assert!(first.is_ok(), "the first claim stands");
        let again = std::thread::Builder::new()
            .name("first::test".into())
            .spawn(|| claim("tuning"))
            .unwrap()
            .join();
        assert!(again.is_ok(), "the same test may claim twice");
        let second = std::thread::Builder::new()
            .name("second::test".into())
            .spawn(|| claim("tuning"))
            .unwrap()
            .join();
        let err = second.expect_err("a second test is refused");
        let msg = err.downcast_ref::<String>().cloned().unwrap_or_default();
        assert!(msg.contains("first::test") && msg.contains("second::test"), "{msg}");
        let unnamed = std::thread::spawn(|| claim("tuning")).join();
        assert!(unnamed.is_ok(), "an unnamed thread claims nothing");
    }
}
