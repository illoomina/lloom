//! The front-ends' own defaults: the keys `lloom chat`, `lloom serve` and
//! `lloom bench` each resolve themselves when no option layer set them, in
//! one table rather than as literals at each front-end.
//!
//! [`ROWS`] is what each front-end resolved before the option tree, and
//! stays what it resolves: a default moves only through an [`Exception`],
//! the maintainer's decision by number, which names the value before and
//! after and whether the change has landed. The schema, `lloom options`,
//! the flag help and `lloom config --frontend` all read this table.
//!
//! A row applies after every layer, to a key no layer named (an optional
//! key a layer set to `none` is unset, so it takes the row too). Nothing
//! here runs per token: it is part of resolving the options, once.

use crate::option_tree::index;
use crate::EngineOptions;
use std::collections::BTreeSet;

/// A front-end of the `lloom` binary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Frontend {
    Chat,
    Serve,
    Bench,
}

impl Frontend {
    pub const ALL: [Frontend; 3] = [Frontend::Chat, Frontend::Serve, Frontend::Bench];

    pub fn name(self) -> &'static str {
        match self {
            Frontend::Chat => "chat",
            Frontend::Serve => "serve",
            Frontend::Bench => "bench",
        }
    }

    /// One of `chat`, `serve`, `bench`, any case.
    pub fn parse(v: &str) -> Result<Frontend, String> {
        let names: Vec<&str> = Frontend::ALL.iter().map(|f| f.name()).collect();
        let arm = lloom_tensor::options::parse_arm(v, &names)?;
        Ok(*Frontend::ALL
            .iter()
            .find(|f| f.name() == arm)
            .expect("an arm is a name"))
    }

    /// Whether this front-end reuses the cached prefix where
    /// `context.prefix.reuse` is unset.
    pub fn reuse_prefix(self) -> bool {
        lloom_tensor::options::parse_bool(value(
            row("context.prefix.reuse", self).expect("every front-end has a row"),
        ))
        .expect("the row is a bool")
    }
}

/// The condition beside "no layer set the key" under which a row applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum When {
    Always,
    /// At a batch: `scheduler.max_sequences` above 1.
    Batched,
    /// The model's family takes the `model` axes (LFM2's `Config`; no
    /// `--model` = the LFM2 snapshot), so a default axis never refuses a
    /// family at load.
    ModelAxes,
}

impl When {
    pub fn holds(self, o: &EngineOptions) -> bool {
        match self {
            When::Always => true,
            When::Batched => o.scheduler.max_sequences > 1,
            // LFM2's detection (`app::family::Lfm2Family`, first in order):
            // a directory holding `config.json`; the GGUF families take none.
            When::ModelAxes => o
                .load
                .model
                .as_deref()
                .map_or(true, |p| p.is_dir() && p.join("config.json").is_file()),
        }
    }

    pub fn describe(self) -> &'static str {
        match self {
            When::Always => "",
            When::Batched => "scheduler.max_sequences > 1",
            When::ModelAxes => "the model's family takes the model axes (LFM2)",
        }
    }
}

/// A key one front-end resolves itself, and the value it resolved before
/// the option tree (in the key's spelling).
#[derive(Debug, Clone, Copy)]
pub struct Row {
    pub key: &'static str,
    pub frontend: Frontend,
    pub when: When,
    pub before: &'static str,
}

/// The same `key = value` row for chat and for serve (the LFM2 decode
/// flow both run by default).
macro_rules! rows {
    ([$($when:ident: $key:literal = $value:literal),* $(,)?] $($row:expr),* $(,)?) => {
        pub const ROWS: &[Row] = &[
            $(
                Row { key: $key, frontend: Frontend::Chat, when: When::$when, before: $value },
                Row { key: $key, frontend: Frontend::Serve, when: When::$when, before: $value },
            )*
            $($row),*
        ];
    };
}

// Every front-end default, each `before` the value its front-end resolved
// before the option tree (the sites read: `decode-driver-design.md`
// Stage 7).
rows! {
    // The tuned decode flow as chat's and serve's default; any layer that
    // names a key keeps its own value. No speculation by default.
    // Every family: the Metal backend arms and the non-blocking ring
    // (GPU-batched EOS, the adaptive commit, the stream reader).
    [
        Always: "backend.metal.command_buffer.api" = "metal4",
        Always: "backend.metal.compiler.merged_unit" = "true",
        Always: "backend.metal.copy2d_compute" = "true",
        Always: "backend.metal.heaps.transient" = "true",
        Always: "decode.readback.kind" = "ring",
        Always: "decode.readback.eos" = "gpu-batched-nonblocking",
        Always: "decode.readback.stream_reader" = "true",
        Always: "prefill.tile" = "64",
        // LFM2 (safetensors) only: its `Config` axes and LFM2 kernels.
        ModelAxes: "kernels.attention.flash.partial_block" = "8",
        // Shared with Qwen4-exp's HC norm arm, +6.6% ms/token there.
        ModelAxes: "kernels.fusion.norm_in_gemv" = "true",
        ModelAxes: "models.lfm2.rows_kernels" = "per-row",
        ModelAxes: "speculative.dspark.context_kv" = "paged",
        ModelAxes: "context.kv.layout" = "paged",
        ModelAxes: "kernels.attention.impl" = "flash",
        ModelAxes: "kernels.attention.qkv_layout" = "fused",
        ModelAxes: "kernels.fusion.conv" = "true",
        ModelAxes: "models.lfm2.conv_prefill_context" = "true",
    ]
    Row {
        key: "context.prefix.reuse",
        frontend: Frontend::Chat,
        when: When::Always,
        before: "true",
    },
    Row {
        key: "context.prefix.reuse",
        frontend: Frontend::Serve,
        when: When::Always,
        before: "false",
    },
    Row {
        key: "context.prefix.reuse",
        frontend: Frontend::Bench,
        when: When::Always,
        before: "false",
    },
    Row {
        key: "generation.max_tokens",
        frontend: Frontend::Chat,
        when: When::Always,
        before: "1024",
    },
    Row {
        key: "scheduler.prefill_chunk",
        frontend: Frontend::Serve,
        when: When::Batched,
        before: "512",
    },
    Row {
        key: "scheduler.prefill_chunk",
        frontend: Frontend::Bench,
        when: When::Batched,
        before: "512",
    },
}

/// A default changed on purpose: the maintainer's decision by number, the
/// key and front-end it moves, the value before and after, and whether the
/// change has landed (until then the front-end resolves `before`).
#[derive(Debug, Clone, Copy)]
pub struct Exception {
    pub decision: u32,
    pub key: &'static str,
    pub frontend: Frontend,
    pub before: &'static str,
    pub after: &'static str,
    pub landed: bool,
    /// Who lands it.
    pub owner: &'static str,
    pub what: &'static str,
}

/// The approved default changes. Landing one is setting its `landed`.
pub const EXCEPTIONS: &[Exception] = &[Exception {
    decision: 36,
    key: "context.prefix.reuse",
    frontend: Frontend::Serve,
    before: "false",
    after: "true",
    landed: true,
    owner: "#946",
    what:
        "serve reuses the cached prefix by default; --no-reuse opts out; the bench cells stay cold",
}];

/// The row of `key` for `frontend`, if the front-end resolves it.
pub fn row(key: &str, frontend: Frontend) -> Option<&'static Row> {
    ROWS.iter().find(|r| r.key == key && r.frontend == frontend)
}

/// The exception on a row, if one names it.
pub fn exception(r: &Row) -> Option<&'static Exception> {
    EXCEPTIONS
        .iter()
        .find(|e| e.key == r.key && e.frontend == r.frontend)
}

/// The value a row resolves to now: its landed exception's, else `before`.
pub fn value(r: &Row) -> &'static str {
    match exception(r) {
        Some(e) if e.landed => e.after,
        _ => r.before,
    }
}

/// The rows of one key, for the schema and `lloom options`.
pub fn rows_of(key: &str) -> impl Iterator<Item = &'static Row> + '_ {
    ROWS.iter().filter(move |r| r.key == key)
}

/// `chat: true, serve: false (36: true, pending), …` for a key's help.
pub fn describe(key: &str) -> Option<String> {
    let parts: Vec<String> = rows_of(key)
        .map(|r| {
            let when = match r.when.describe() {
                "" => String::new(),
                w => format!(" when {w}"),
            };
            let change = match exception(r) {
                Some(e) if e.landed => format!(" (decision {}: was {})", e.decision, e.before),
                Some(e) => format!(" (decision {}: {} once landed)", e.decision, e.after),
                None => String::new(),
            };
            format!("{}: {}{when}{change}", r.frontend.name(), value(r))
        })
        .collect();
    (!parts.is_empty()).then(|| parts.join(", "))
}

/// Apply `frontend`'s rows to the keys of `o` no layer named (`explicit`,
/// by key path) and whose condition holds; an optional key at `null` is
/// unset whoever wrote it. Returns the keys it set.
pub fn apply(
    o: &mut EngineOptions,
    frontend: Frontend,
    explicit: &BTreeSet<String>,
) -> Result<Vec<&'static str>, String> {
    let idx = index();
    let mut set = Vec::new();
    for r in ROWS.iter().filter(|r| r.frontend == frontend) {
        if !r.when.holds(o) {
            continue;
        }
        let unset = idx.flat(o).get(r.key).is_some_and(|v| v.is_null());
        if explicit.contains(r.key) && !unset {
            continue;
        }
        idx.set(o, r.key, value(r))
            .map_err(|e| format!("the {} default of {}: {e}", frontend.name(), r.key))?;
        set.push(r.key);
    }
    Ok(set)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    /// Every row names a key of the tree whose value it can be, and every
    /// exception names a row and agrees with it on `before`.
    #[test]
    fn every_row_is_a_key_and_every_exception_a_row() {
        let idx = index();
        for r in ROWS {
            assert!(idx.keys.contains_key(r.key), "{} is not a key", r.key);
            let mut o = EngineOptions::default();
            idx.set(&mut o, r.key, r.before)
                .unwrap_or_else(|e| panic!("{} = {}: {e}", r.key, r.before));
        }
        for e in EXCEPTIONS {
            let r = row(e.key, e.frontend)
                .unwrap_or_else(|| panic!("decision {} names no row", e.decision));
            assert_eq!(r.before, e.before, "decision {}", e.decision);
            assert_ne!(e.before, e.after, "decision {}", e.decision);
        }
    }

    /// The unset defaults, per front-end: what each front-end resolved
    /// before the option tree, except where a landed exception says
    /// otherwise; a pending exception changes nothing yet.
    #[test]
    fn each_front_end_resolves_its_defaults_and_only_the_named_exceptions_moved() {
        let idx = index();
        for f in Frontend::ALL {
            for batch in [1usize, 4] {
                let mut o = EngineOptions::default();
                o.scheduler.max_sequences = batch;
                apply(&mut o, f, &BTreeSet::new()).unwrap();
                let flat = idx.flat(&o);
                for r in ROWS.iter().filter(|r| r.frontend == f) {
                    let mut want = EngineOptions::default();
                    want.scheduler.max_sequences = batch;
                    if r.when.holds(&want) {
                        let expect = match exception(r) {
                            Some(e) if e.landed => e.after,
                            _ => r.before,
                        };
                        idx.set(&mut want, r.key, expect).unwrap();
                    }
                    assert_eq!(
                        flat[r.key],
                        idx.flat(&want)[r.key],
                        "{} {} at batch {batch}",
                        f.name(),
                        r.key
                    );
                }
            }
        }
        // The rows by value, so a moved row fails here with its name.
        let at = |f, key: &str, batch: usize| -> Value {
            let mut o = EngineOptions::default();
            o.scheduler.max_sequences = batch;
            apply(&mut o, f, &BTreeSet::new()).unwrap();
            idx.flat(&o)[key].clone()
        };
        let decision_36 = EXCEPTIONS.iter().find(|e| e.decision == 36).unwrap();
        assert_eq!(at(Frontend::Chat, "context.prefix.reuse", 1), true);
        assert_eq!(
            at(Frontend::Serve, "context.prefix.reuse", 1),
            decision_36.landed,
            "decision 36 decides serve's reuse"
        );
        assert_eq!(at(Frontend::Bench, "context.prefix.reuse", 1), false);
        assert_eq!(at(Frontend::Chat, "generation.max_tokens", 1), 1024);
        assert_eq!(at(Frontend::Serve, "generation.max_tokens", 1), 512);
        assert_eq!(at(Frontend::Serve, "scheduler.prefill_chunk", 1), 0);
        assert_eq!(at(Frontend::Serve, "scheduler.prefill_chunk", 4), 512);
        assert_eq!(at(Frontend::Bench, "scheduler.prefill_chunk", 4), 512);
        assert_eq!(at(Frontend::Chat, "scheduler.prefill_chunk", 4), 0);
        assert!(!Frontend::Bench.reuse_prefix() && Frontend::Chat.reuse_prefix());
    }

    /// A key a layer named keeps its value, the library default included;
    /// `none` on the optional key leaves it to the front-end.
    #[test]
    fn a_key_a_layer_named_keeps_its_value() {
        let idx = index();
        let named = |k: &str| BTreeSet::from([k.to_string()]);
        let mut o = EngineOptions::default();
        apply(&mut o, Frontend::Chat, &named("generation.max_tokens")).unwrap();
        assert_eq!(o.sampling.max_tokens, 512, "an explicit 512 is not unset");
        let mut o = EngineOptions::default();
        o.scheduler.max_sequences = 4;
        apply(&mut o, Frontend::Serve, &named("scheduler.prefill_chunk")).unwrap();
        assert_eq!(
            o.decode.prefill.chunk, 0,
            "an explicit 0 is the whole-prompt arm"
        );
        let mut o = EngineOptions::default();
        idx.set(&mut o, "context.prefix.reuse", "false").unwrap();
        apply(&mut o, Frontend::Chat, &named("context.prefix.reuse")).unwrap();
        assert_eq!(o.prefix.reuse, Some(false), "an explicit opt-out holds");
        let mut o = EngineOptions::default();
        idx.set(&mut o, "context.prefix.reuse", "none").unwrap();
        apply(&mut o, Frontend::Chat, &named("context.prefix.reuse")).unwrap();
        assert_eq!(o.prefix.reuse, Some(true), "`none` is the front-end's own");
    }

    #[test]
    fn the_description_names_each_front_end_and_the_pending_change() {
        let d = describe("context.prefix.reuse").unwrap();
        assert!(d.starts_with("chat: true, serve: "), "{d}");
        assert!(d.contains("decision 36"), "{d}");
        assert_eq!(
            describe("scheduler.prefill_chunk").unwrap(),
            "serve: 512 when scheduler.max_sequences > 1, bench: 512 when scheduler.max_sequences > 1"
        );
        assert!(describe("decode.in_flight_forwards").is_none());
        assert_eq!(Frontend::parse(" Serve "), Ok(Frontend::Serve));
        assert!(Frontend::parse("repl").is_err());
    }
}
