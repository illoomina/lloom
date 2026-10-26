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
//! key a layer set to `none` is unset, so it takes the row too). A named
//! LFM2 preset keeps the axes it sets (`app::preset_keys`). Nothing here
//! runs per token: it is part of resolving the options, once.

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
    /// On Metal: a build with the backend, not `--cpu`. The ring readback,
    /// its GPU-batched EOS and the Metal backend arms are Metal mechanisms;
    /// a CPU run keeps the library's blocking readback (and its silence).
    Metal,
    /// The model's family takes the `model` axes (LFM2's `Config`; no
    /// `--model` = the LFM2 snapshot), so a default axis never refuses a
    /// family at load.
    ModelAxes,
    /// LFM2's speculative body at one sequence: [`When::ModelAxes`],
    /// `speculative.method = dspark`, `scheduler.max_sequences = 1`, on a
    /// preset whose own KV under a drafter is the in-place one (`auto`,
    /// `rewindable`; `pool-spec` keeps its page pool). Every verify window
    /// is a multi-row pass, and on the page pool such a pass attends over a
    /// copy of the cache gathered out of the pool (`speculation.md`
    /// §10.2z-x).
    DsparkSingle,
    /// [`When::Metal`] at one sequence (`scheduler.max_sequences = 1`): the
    /// serve keep-warm takes the single-sequence request slot, which a
    /// batched server does not have.
    MetalSingle,
    /// [`When::Metal`] at a batch no wider than the row axis's ceiling
    /// (`scheduler.max_sequences` from 2 to `kernels.gemm.f16_rows_max`): the
    /// batched decode step's rows on the GEMV's row axis, a Metal kernel
    /// reached only by the batched loop. Every step of such a batch fits
    /// under the ceiling, so a row's route never depends on how many rows its
    /// steps held, which follows the requests' arrival.
    MetalBatched,
}

impl When {
    pub fn holds(self, o: &EngineOptions) -> bool {
        match self {
            When::Always => true,
            When::Batched => o.scheduler.max_sequences > 1,
            // LFM2's detection (`app::family::Lfm2Family`, first in order):
            // a directory holding `config.json`; the GGUF families take none.
            // The rows are Metal defaults (the paged KV pool is read by a
            // Metal kernel, FlashDecoding, the fused conv): on `--cpu`, or a
            // build without Metal, they would refuse every LFM2 load
            // (`KvAppend::Paged is Metal-only`, 22 engine tests red on
            // 2026-10-19), so the library's own values stay.
            When::Metal => cfg!(feature = "metal") && o.load.device != "cpu",
            When::ModelAxes => {
                When::Metal.holds(o)
                    && o.load.model.as_deref().is_none_or(|p| {
                        p.is_dir() && p.join("config.json").is_file()
                    })
            }
            When::DsparkSingle => {
                When::ModelAxes.holds(o)
                    && o.speculative.method == "dspark"
                    && o.scheduler.max_sequences == 1
                    && matches!(o.model.cache.as_str(), "auto" | "rewindable")
            }
            When::MetalSingle => When::Metal.holds(o) && o.scheduler.max_sequences == 1,
            When::MetalBatched => {
                When::Metal.holds(o)
                    && o.scheduler.max_sequences > 1
                    && o.scheduler.max_sequences <= o.tuning.gemm.f16_rows_max
            }
        }
    }

    pub fn describe(self) -> &'static str {
        match self {
            When::Always => "",
            When::Batched => "scheduler.max_sequences > 1",
            When::Metal => "on Metal (a Metal build, not --cpu)",
            When::ModelAxes => "the model's family takes the model axes (LFM2), on Metal",
            When::DsparkSingle => {
                "speculative.method = dspark at scheduler.max_sequences = 1 on the auto or rewindable preset (LFM2, on Metal)"
            }
            When::MetalSingle => "on Metal at scheduler.max_sequences = 1",
            When::MetalBatched => {
                "on Metal at scheduler.max_sequences from 2 to kernels.gemm.f16_rows_max"
            }
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
    // Every family on Metal: the Metal backend arms and the non-blocking
    // ring (GPU-batched EOS, the adaptive commit, the stream reader) —
    // Qwen4-exp included, by the maintainer's choice (2026-10-19): the ring
    // reads each fed id back there without a PLE today (the load warning
    // says so, #829), and the cadence work planned is to make the
    // non-blocking ring the best arm for every family.
    [
        Metal: "backend.metal.command_buffer.api" = "metal4",
        Metal: "backend.metal.compiler.merged_unit" = "true",
        Metal: "backend.metal.copy2d_compute" = "true",
        Metal: "backend.metal.heaps.transient" = "true",
        Metal: "decode.readback.kind" = "ring",
        Metal: "decode.readback.eos" = "gpu-batched-nonblocking",
        Metal: "decode.readback.stream_reader" = "true",
        // The turn-reuse default — LGD (decisions 35/43/44, maintainer
        // 2026-10-19, from #993 run 3's comparison, PR1064): the EOS gate
        // turns a warm turn's restore into an extend (Qwen3.6-27B warm TTFT
        // 1.14 s → 0.44 s, 35B-A3B 336 → 175 ms), the shallower lead and the
        // gate's cores arm cut the reply-end latency (27B 1160 → 150–280 ms);
        // the per-token rate was unresolved on every arm. The gate keys are
        // Qwen3.5's (no other family reads them); the lead is the ring's, so
        // every family on Metal takes it — one default authority, to be
        // confirmed on LFM2 by #859's screen. Each is an explicit switch for
        // an A/B: `--qwen3-5-eos-gate=false`, `--qwen3-5-eos-gate-cores=false`,
        // `--set decode.readback.queued_steps=0` (the batching's lead).
        // `before` is the library's value; decisions 43/35 (`EXCEPTIONS`)
        // land the new one, as decision 36 did for serve's reuse.
        Metal: "models.qwen3_5.eos_gate" = "false",
        Metal: "models.qwen3_5.eos_gate_cores" = "false",
        Metal: "decode.readback.queued_steps" = "0",
        // The device residency set off the Metal 3 queue (`tuning.md`
        // §6.3n): `before` is the library's value; decision 56
        // (`EXCEPTIONS`) lands `heartbeat`. Metal 4 ignores the arm.
        Metal: "kernels.residency.attach" = "queue",
        // LFM2 (safetensors) only: its `Config` axes and LFM2 kernels.
        ModelAxes: "kernels.attention.flash.partial_block" = "8",
        // Shared with Qwen4-exp's HC norm arm, +6.6% ms/token there.
        ModelAxes: "kernels.fusion.norm_in_gemv" = "true",
        ModelAxes: "models.lfm2.rows_kernels" = "per-row",
        ModelAxes: "speculative.dspark.context_kv" = "paged",
        ModelAxes: "context.kv.layout" = "paged",
        // LFM2's speculative body at one sequence: `before` is the paged
        // KV the row above gave it; decision 52 (`EXCEPTIONS`) keeps its
        // preset's in-place KV, because a verify window is a multi-row
        // pass, which on the page pool attends over a copy of the cache
        // gathered every round (`speculation.md` §10.2z-x). After the
        // paged row, which it overrides where both hold.
        DsparkSingle: "context.kv.layout" = "paged",
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
    // Serve keeps the GPU's memory wired while it idles, every second
    // (under the 2–3 s after which the OS unwires an idle residency set),
    // for `server.keep_warm_for_s` after the last request
    // (`machine-model.md` §3.4d).
    Row {
        key: "server.keep_warm",
        frontend: Frontend::Serve,
        when: When::MetalSingle,
        before: "1000",
    },
    // Where that keep-warm runs, it stands down under memory pressure
    // (`tuning.md` §6.3n): `before` is the library's value; decision 57
    // (`EXCEPTIONS`) lands the stop.
    Row {
        key: "server.keep_warm_pressure",
        frontend: Frontend::Serve,
        when: When::MetalSingle,
        before: "false",
    },
    // The batched decode step's rows on the GEMV's row axis under `fast`
    // (`tuning.md` §13.5zb): `before` is the library's value; decision 54
    // (`EXCEPTIONS`) lands the new one where a batched loop runs on Metal.
    Row {
        key: "kernels.gemm.f16_rows_decode",
        frontend: Frontend::Serve,
        when: When::MetalBatched,
        before: "false",
    },
    Row {
        key: "kernels.gemm.f16_rows_decode",
        frontend: Frontend::Bench,
        when: When::MetalBatched,
        before: "false",
    },
}

/// A default changed on purpose: the maintainer's decision by number, the
/// row it moves (key, front-end and condition), the value before and after,
/// and whether the change has landed (until then the front-end resolves
/// `before`).
#[derive(Debug, Clone, Copy)]
pub struct Exception {
    pub decision: u32,
    pub key: &'static str,
    pub frontend: Frontend,
    /// The row's condition: a key may carry a row per condition.
    pub when: When,
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
    when: When::Always,
    before: "false",
    after: "true",
    landed: true,
    owner: "#946",
    what:
        "serve reuses the cached prefix by default; --no-reuse opts out; the bench cells stay cold",
}, Exception {
    decision: 43,
    key: "models.qwen3_5.eos_gate",
    frontend: Frontend::Chat,
    when: When::Metal,
    before: "false",
    after: "true",
    landed: true,
    owner: "#993",
    what: "the EOS gate on by default on Metal (LGD): a warm turn extends instead of restoring; --qwen3-5-eos-gate=false opts out",
}, Exception {
    decision: 43,
    key: "models.qwen3_5.eos_gate",
    frontend: Frontend::Serve,
    when: When::Metal,
    before: "false",
    after: "true",
    landed: true,
    owner: "#993",
    what: "the EOS gate on by default on Metal (LGD): a warm turn extends instead of restoring; --qwen3-5-eos-gate=false opts out",
}, Exception {
    decision: 43,
    key: "models.qwen3_5.eos_gate_cores",
    frontend: Frontend::Chat,
    when: When::Metal,
    before: "false",
    after: "true",
    landed: true,
    owner: "#993",
    what: "the gate's cores arm on by default on Metal (LGD's D): dead post-terminal compute skipped; --qwen3-5-eos-gate-cores=false opts out",
}, Exception {
    decision: 43,
    key: "models.qwen3_5.eos_gate_cores",
    frontend: Frontend::Serve,
    when: When::Metal,
    before: "false",
    after: "true",
    landed: true,
    owner: "#993",
    what: "the gate's cores arm on by default on Metal (LGD's D): dead post-terminal compute skipped; --qwen3-5-eos-gate-cores=false opts out",
}, Exception {
    decision: 35,
    key: "decode.readback.queued_steps",
    frontend: Frontend::Chat,
    when: When::Metal,
    before: "0",
    after: "4",
    landed: true,
    owner: "#993",
    what: "the lead's queued part is 4 steps on Metal (LGD's L): reply-end latency 1160 -> 280 ms on the 27B; 0 = the batching's lead",
}, Exception {
    decision: 35,
    key: "decode.readback.queued_steps",
    frontend: Frontend::Serve,
    when: When::Metal,
    before: "0",
    after: "4",
    landed: true,
    owner: "#993",
    what: "the lead's queued part is 4 steps on Metal (LGD's L): reply-end latency 1160 -> 280 ms on the 27B; 0 = the batching's lead",
}, Exception {
    decision: 52,
    key: "context.kv.layout",
    frontend: Frontend::Chat,
    when: When::DsparkSingle,
    before: "paged",
    after: "in-place",
    landed: true,
    owner: "#1068",
    what: "the speculative body at one sequence keeps its preset's in-place KV on Metal: the verify window reads the slab instead of a copy gathered out of the page pool every round; --kv-append paged opts out",
}, Exception {
    decision: 52,
    key: "context.kv.layout",
    frontend: Frontend::Serve,
    when: When::DsparkSingle,
    before: "paged",
    after: "in-place",
    landed: true,
    owner: "#1068",
    what: "the speculative body at one sequence keeps its preset's in-place KV on Metal: the verify window reads the slab instead of a copy gathered out of the page pool every round; --kv-append paged opts out",
}, Exception {
    decision: 54,
    key: "kernels.gemm.f16_rows_decode",
    frontend: Frontend::Serve,
    when: When::MetalBatched,
    before: "false",
    after: "true",
    landed: true,
    owner: "#1084",
    what: "at a batch of 2 to kernels.gemm.f16_rows_max sequences, a batched step's live rows take the GEMV's row axis under fast, a step with prompt pieces included (deterministic never does); --gemm-f16-rows-decode=false opts out",
}, Exception {
    decision: 54,
    key: "kernels.gemm.f16_rows_decode",
    frontend: Frontend::Bench,
    when: When::MetalBatched,
    before: "false",
    after: "true",
    landed: true,
    owner: "#1084",
    what: "at a batch of 2 to kernels.gemm.f16_rows_max sequences, a batched step's live rows take the GEMV's row axis under fast, a step with prompt pieces included (deterministic never does); --gemm-f16-rows-decode=false opts out",
}, Exception {
    decision: 56,
    key: "kernels.residency.attach",
    frontend: Frontend::Chat,
    when: When::Metal,
    before: "queue",
    after: "heartbeat",
    landed: true,
    owner: "#1092",
    what: "the device residency set is not on the Metal 3 queue, so a command buffer waits only for what it binds; Metal 4 ignores the arm (its queue binds through per-slot sets); --residency-attach queue opts out",
}, Exception {
    decision: 56,
    key: "kernels.residency.attach",
    frontend: Frontend::Serve,
    when: When::Metal,
    before: "queue",
    after: "heartbeat",
    landed: true,
    owner: "#1092",
    what: "the device residency set is not on the Metal 3 queue, so a command buffer waits only for what it binds; Metal 4 ignores the arm (its queue binds through per-slot sets); --residency-attach queue opts out",
}, Exception {
    decision: 57,
    key: "server.keep_warm_pressure",
    frontend: Frontend::Serve,
    when: When::MetalSingle,
    before: "false",
    after: "true",
    landed: true,
    owner: "#1092",
    what: "the keep-warm reads the memory-pressure level each beat: at warn or critical it stops requesting and ends the set's residency, at normal it resumes; --keep-warm-pressure=false opts out",
}];

/// The first row of `key` for `frontend`, if the front-end resolves it.
pub fn row(key: &str, frontend: Frontend) -> Option<&'static Row> {
    rows_for(key, frontend).next()
}

/// The rows of `key` for `frontend`, in the order [`apply`] takes them: a
/// later row whose condition holds overrides an earlier one.
pub fn rows_for(key: &str, frontend: Frontend) -> impl Iterator<Item = &'static Row> + '_ {
    ROWS.iter()
        .filter(move |r| r.key == key && r.frontend == frontend)
}

/// The value `frontend`'s rows give `key` on `o`, where no layer named it:
/// the last row whose condition holds, `None` where none holds.
pub fn resolved(key: &str, frontend: Frontend, o: &EngineOptions) -> Option<&'static str> {
    rows_for(key, frontend)
        .filter(|r| r.when.holds(o))
        .last()
        .map(value)
}

/// The exception on a row, if one names it.
pub fn exception(r: &Row) -> Option<&'static Exception> {
    EXCEPTIONS
        .iter()
        .find(|e| e.key == r.key && e.frontend == r.frontend && e.when == r.when)
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
/// unset whoever wrote it. A key a named preset sets (`preset`) is the
/// preset's, at `null` too. Returns the keys it set.
pub fn apply(
    o: &mut EngineOptions,
    frontend: Frontend,
    explicit: &BTreeSet<String>,
    preset: &BTreeSet<String>,
) -> Result<Vec<&'static str>, String> {
    let idx = index();
    let mut set = Vec::new();
    for r in ROWS.iter().filter(|r| r.frontend == frontend) {
        if !r.when.holds(o) || preset.contains(r.key) {
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
            let named: Vec<&Row> = rows_for(e.key, e.frontend)
                .filter(|r| r.when == e.when)
                .collect();
            assert_eq!(named.len(), 1, "decision {} names one row", e.decision);
            assert_eq!(named[0].before, e.before, "decision {}", e.decision);
            assert_ne!(e.before, e.after, "decision {}", e.decision);
            assert_eq!(
                exception(named[0]).map(|x| (x.decision, x.frontend, x.when)),
                Some((e.decision, e.frontend, e.when)),
                "decision {} is its row's exception",
                e.decision
            );
        }
    }

    /// The unset defaults, per front-end: what each front-end resolved
    /// before the option tree, except where a landed exception says
    /// otherwise; a pending exception changes nothing yet. A key with two
    /// rows takes the last whose condition holds.
    #[test]
    fn each_front_end_resolves_its_defaults_and_only_the_named_exceptions_moved() {
        let idx = index();
        for f in Frontend::ALL {
            for (batch, method) in [(1usize, "none"), (4, "none"), (1, "dspark"), (4, "dspark")] {
                let base = || {
                    let mut o = EngineOptions::default();
                    o.scheduler.max_sequences = batch;
                    o.speculative.method = method.into();
                    o
                };
                let mut o = base();
                apply(&mut o, f, &BTreeSet::new(), &BTreeSet::new()).unwrap();
                let flat = idx.flat(&o);
                for r in ROWS.iter().filter(|r| r.frontend == f) {
                    let mut want = base();
                    let held: Vec<&Row> =
                        rows_for(r.key, f).filter(|r| r.when.holds(&want)).collect();
                    if let Some(last) = held.last() {
                        let expect = match exception(last) {
                            Some(e) if e.landed => e.after,
                            _ => last.before,
                        };
                        assert_eq!(resolved(r.key, f, &want), Some(expect));
                        idx.set(&mut want, r.key, expect).unwrap();
                    } else {
                        assert_eq!(resolved(r.key, f, &want), None);
                    }
                    assert_eq!(
                        flat[r.key],
                        idx.flat(&want)[r.key],
                        "{} {} at batch {batch}, speculative.method {method}",
                        f.name(),
                        r.key
                    );
                }
            }
        }
        // The speculative body at one sequence keeps the in-place KV on
        // chat and serve; at a batch, without a drafter, on the pool preset
        // and on bench, the paged row (or none) holds.
        let kv = |f, batch: usize, method: &str, cache: &str| -> Value {
            let mut o = EngineOptions::default();
            o.scheduler.max_sequences = batch;
            o.speculative.method = method.into();
            o.model.cache = cache.into();
            apply(&mut o, f, &BTreeSet::new(), &BTreeSet::new()).unwrap();
            idx.flat(&o)["context.kv.layout"].clone()
        };
        let library = idx.flat(&EngineOptions::default())["context.kv.layout"].clone();
        let metal = When::ModelAxes.holds(&EngineOptions::default());
        let typed = |v: &str| {
            let mut o = EngineOptions::default();
            idx.set(&mut o, "context.kv.layout", v).unwrap();
            idx.flat(&o)["context.kv.layout"].clone()
        };
        let (in_place, paged) = if metal {
            (typed("in-place"), typed("paged"))
        } else {
            (library.clone(), library.clone())
        };
        for f in [Frontend::Chat, Frontend::Serve] {
            assert_eq!(kv(f, 1, "dspark", "auto"), in_place, "{}", f.name());
            assert_eq!(kv(f, 4, "dspark", "auto"), paged, "{} at a batch", f.name());
            assert_eq!(kv(f, 1, "none", "auto"), paged, "{} plain", f.name());
            assert_eq!(kv(f, 1, "auto", "auto"), paged, "{} method auto", f.name());
            assert_eq!(
                kv(f, 1, "dspark", "rewindable"),
                in_place,
                "{} rewindable",
                f.name()
            );
            assert_eq!(
                kv(f, 1, "dspark", "pool-spec"),
                paged,
                "{} pool-spec",
                f.name()
            );
        }
        assert_eq!(kv(Frontend::Bench, 1, "dspark", "auto"), library);
        // The rows by value, so a moved row fails here with its name.
        let at = |f, key: &str, batch: usize| -> Value {
            let mut o = EngineOptions::default();
            o.scheduler.max_sequences = batch;
            apply(&mut o, f, &BTreeSet::new(), &BTreeSet::new()).unwrap();
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
        let none = BTreeSet::new();
        let mut o = EngineOptions::default();
        apply(
            &mut o,
            Frontend::Chat,
            &named("generation.max_tokens"),
            &none,
        )
        .unwrap();
        assert_eq!(o.sampling.max_tokens, 512, "an explicit 512 is not unset");
        let mut o = EngineOptions::default();
        o.scheduler.max_sequences = 4;
        apply(
            &mut o,
            Frontend::Serve,
            &named("scheduler.prefill_chunk"),
            &none,
        )
        .unwrap();
        assert_eq!(
            o.decode.prefill.chunk, 0,
            "an explicit 0 is the whole-prompt arm"
        );
        let mut o = EngineOptions::default();
        idx.set(&mut o, "context.prefix.reuse", "false").unwrap();
        apply(
            &mut o,
            Frontend::Chat,
            &named("context.prefix.reuse"),
            &none,
        )
        .unwrap();
        assert_eq!(o.prefix.reuse, Some(false), "an explicit opt-out holds");
        let mut o = EngineOptions::default();
        idx.set(&mut o, "context.prefix.reuse", "none").unwrap();
        apply(
            &mut o,
            Frontend::Chat,
            &named("context.prefix.reuse"),
            &none,
        )
        .unwrap();
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
