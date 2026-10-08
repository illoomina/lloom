//! The couplings between options, as data: where one option changes what
//! another does. Each names its kind, its keys and the code that implements
//! it; the statically checkable refusals are also schema rules and are
//! refused when the options resolve, by id, before anything loads.
//!
//! Kinds: `resolves` (an `auto` decided by another option, reported),
//! `clamps` (asked for, turned off, with a load note), `implies` (sets
//! another), `refuses` (an error), `ignored` (no effect unless …).

use crate::options::EngineOptions;
use lloom::generation::Sampling;
use serde_json::{json, Value};

#[derive(Debug, Clone, Copy)]
pub struct Coupling {
    pub id: &'static str,
    pub kind: &'static str,
    pub when: &'static str,
    pub effect: &'static str,
    pub keys: &'static [&'static str],
    /// The code that implements it (`path:line`, the function).
    pub site: &'static str,
    /// Checked against the config before anything loads (and a schema rule).
    pub is_static: bool,
}

pub const COUPLINGS: &[Coupling] = &[
    Coupling {
        id: "C1",
        kind: "resolves",
        when: "decode.readback.kind = auto, and the sampler is host-side (top-k or top-p at a temperature, no device sampler)",
        effect: "the run takes the blocking readback, reported on the turn line",
        keys: &["decode.readback.kind", "generation.top_k", "generation.top_p", "generation.temperature", "generation.device_sampler"],
        site: "lloom-engine/src/decode.rs ReadbackArm::{ring_admits, resolve}",
        is_static: false,
    },
    Coupling {
        id: "C2",
        kind: "refuses",
        when: "decode.readback.kind = ring with the host-side sampler of C1",
        effect: "refused: the ring has no host round trip for the draw; `auto` makes the choice instead",
        keys: &["decode.readback.kind", "generation.top_k", "generation.top_p", "generation.temperature", "generation.device_sampler"],
        site: "lloom/src/generation/readback.rs Ring::check_sampler; at resolve: couplings::refuse",
        is_static: true,
    },
    Coupling {
        id: "C3",
        kind: "ignored",
        when: "generation.temperature unset or below 1e-7",
        effect: "top_k, top_p and the device sampler have no effect: argmax runs",
        keys: &["generation.top_k", "generation.top_p", "generation.device_sampler", "generation.temperature"],
        site: "lloom-engine/src/decode.rs DecodeParams::sampling_of",
        is_static: false,
    },
    Coupling {
        id: "C4",
        kind: "clamps",
        when: "decode.readback.adaptive_commit, on a ring arm with lookahead_forwards != 1 or eos = blocking",
        effect: "the adaptive commit is off and the ring runs the fixed cadence; a load note says why",
        keys: &["decode.readback.adaptive_commit", "decode.readback.lookahead_forwards", "decode.readback.eos"],
        site: "lloom-engine/src/decode.rs RingArm::why_not_adaptive; options.rs ReadbackConfig::adaptive_clamp",
        is_static: false,
    },
    Coupling {
        id: "C5",
        kind: "clamps",
        when: "the adaptive commit does not run (C4, or the blocking arm)",
        effect: "the EOS gate is off (it rides the adaptive commit)",
        keys: &["decode.readback.eos_gate", "decode.readback.adaptive_commit"],
        site: "lloom-engine/src/decode.rs (opts.eos_gate = eos_gate && adaptive_commit)",
        is_static: false,
    },
    Coupling {
        id: "C6",
        kind: "clamps",
        when: "the loaded family's decode kernels carry no gated variant (Qwen4-exp)",
        effect: "the EOS gate is off, with a note where it would have run",
        keys: &["decode.readback.eos_gate"],
        site: "lloom-engine/src/app/family.rs clamp_ungated_eos_gate",
        is_static: false,
    },
    Coupling {
        id: "C7",
        kind: "clamps",
        when: "executor.replay.enabled",
        effect: "the EOS gate is off, with a load note (the gate reaches directly encoded dispatches only)",
        keys: &["decode.readback.eos_gate", "executor.replay.enabled"],
        site: "lloom-engine/src/app/mod.rs clamp_eos_gate; options.rs ExecutorConfig::refuses_eos_gate",
        is_static: false,
    },
    Coupling {
        id: "C8",
        kind: "implies",
        when: "executor.replay.enabled",
        effect: "the packed dispatches and the constants pool, and the arena (not under a speculative body)",
        keys: &["executor.replay.enabled", "executor.packed", "executor.arena.enabled"],
        site: "lloom-engine/src/app/common.rs install_executor",
        is_static: false,
    },
    Coupling {
        id: "C9",
        kind: "implies",
        when: "the adaptive commit runs",
        effect: "the tokens stream off the ring reader (decode.readback.stream_reader)",
        keys: &["decode.readback.stream_reader", "decode.readback.adaptive_commit"],
        site: "lloom-engine/src/decode.rs (reader = adaptive || stream_reader)",
        is_static: false,
    },
    Coupling {
        id: "C10",
        kind: "refuses",
        when: "decode.readback.lookahead_forwards > 1",
        effect: "refused: the loop runs at most one forward past an unlearned EOS",
        keys: &["decode.readback.lookahead_forwards"],
        site: "lloom-engine/src/options.rs ReadbackConfig::check",
        is_static: true,
    },
    Coupling {
        id: "C11",
        kind: "refuses",
        when: "speculative.method is a drafter, and a host-side sampler (top-k / top-p), a logits processor or a thinking budget is set; or the drafter is mtp / dflash and a temperature is set",
        effect: "refused: a verify window draws each position with position-keyed noise (dspark serves argmax, plain temperature and --gumbel), which a host sort cannot take and a processor's history would be corrupted by; the mtp and dflash targets' verifies read an argmax back and draw nothing",
        keys: &["speculative.method", "generation.temperature", "generation.top_k", "generation.top_p", "generation.min_p", "generation.repetition_penalty", "generation.presence_penalty", "generation.frequency_penalty", "generation.logit_bias", "reasoning.budget"],
        site: "lloom-engine/src/decode.rs DecodeSession::decode_speculative_until; at resolve and after `auto` resolves: couplings::refuse",
        is_static: true,
    },
    Coupling {
        id: "C12",
        kind: "refuses",
        when: "scheduler.max_sequences > 1, and a logits processor or a thinking budget is set",
        effect: "refused: the batched driver's rows draw without processors and feed no forced token (serve them at a batch of 1)",
        keys: &["scheduler.max_sequences", "generation.min_p", "generation.repetition_penalty", "generation.presence_penalty", "generation.frequency_penalty", "generation.logit_bias", "reasoning.budget"],
        site: "lloom-engine/src/batch.rs Scheduler::prepare; at resolve: couplings::refuse",
        is_static: true,
    },
    Coupling {
        id: "C13",
        kind: "ignored",
        when: "decode.readback.eos != poll",
        effect: "decode.readback.wait has no effect",
        keys: &["decode.readback.wait", "decode.readback.eos"],
        site: "lloom-engine/src/options.rs EosKind::eos_read",
        is_static: false,
    },
    Coupling {
        id: "C14",
        kind: "ignored",
        when: "decode.readback.eos is not host-drain, gpu-batched or gpu-batched-nonblocking",
        effect: "decode.readback.wake_every_commits has no effect (only the deferred reads write the commit marker it paces)",
        keys: &["decode.readback.wake_every_commits", "decode.readback.eos"],
        site: "lloom/src/generation/readback.rs Ring::commit (the deferred-EOS marker)",
        is_static: false,
    },
    Coupling {
        id: "C15",
        kind: "ignored",
        when: "the adaptive commit does not run; for the fused publish also a sampler that is not argmax without processors",
        effect: "decode.readback.run_ahead_steps and decode.readback.fused_publish have no effect",
        keys: &["decode.readback.run_ahead_steps", "decode.readback.fused_publish", "decode.readback.adaptive_commit"],
        site: "lloom/src/nn/decode.rs (set_fused_publish(adaptive && fused); the run-ahead bound under adaptive_commit); lloom-engine/src/decode.rs (fused = argmax, no processors)",
        is_static: false,
    },
    Coupling {
        id: "C16",
        kind: "refuses",
        when: "kernels.attention.flash.pages_per_chunk set, on Qwen4-exp, or on LFM2 off the FlashDecoding attention",
        effect: "refused at load by name",
        keys: &["kernels.attention.flash.pages_per_chunk", "kernels.attention.impl"],
        site: "lloom-engine/src/app/family.rs Qwen4ExpFamily::resolve; app/lfm2.rs Engine::load_as",
        is_static: false,
    },
    Coupling {
        id: "C17",
        kind: "resolves",
        when: "speculative.method is a drafter (LFM2: dspark)",
        effect: "models.lfm2.preset = auto takes the rewindable shape (pool-spec at a batch); the arena is not recorded under the speculative body and the replay executor is refused",
        keys: &["speculative.method", "models.lfm2.preset", "executor.arena.enabled", "executor.replay.enabled"],
        site: "lloom-engine/src/app/lfm2.rs apply_preset; app/common.rs install_executor",
        is_static: false,
    },
    Coupling {
        id: "C18",
        kind: "resolves",
        when: "scheduler.max_sequences > 1",
        effect: "scheduler.rows = auto takes the pool and models.lfm2.preset = auto the pool preset (paged KV)",
        keys: &["scheduler.max_sequences", "scheduler.rows", "models.lfm2.preset", "context.kv.layout"],
        site: "lloom-engine/src/app/lfm2.rs RowsArm::parse, apply_preset; lloom/src/models/lfm2_presets.rs pool",
        is_static: false,
    },
    Coupling {
        id: "C19",
        kind: "resolves",
        when: "executor.arena.enabled or executor.replay.enabled",
        effect: "the pass-plan cache is not installed: the single-key arena records the step instead",
        keys: &["executor.pass_plans", "executor.arena.enabled", "executor.replay.enabled"],
        site: "lloom-engine/src/options.rs ExecutorConfig::pass_plans_armed",
        is_static: false,
    },
];

/// The couplings naming `key`.
pub fn of(key: &str) -> Vec<&'static str> {
    COUPLINGS
        .iter()
        .filter(|c| c.keys.contains(&key))
        .map(|c| c.id)
        .collect()
}

/// The sampler is drawn on the host, which the ring cannot carry.
fn host_side(s: &Sampling) -> bool {
    !matches!(
        s,
        Sampling::ArgMax | Sampling::All { .. } | Sampling::GumbelSoftmax { .. }
    )
}

const DRAFTERS: [&str; 3] = ["dspark", "mtp", "dflash"];
/// The drafters whose target verifies by sampling each window position
/// (`Probes::verify_and_probe_sampled`): LFM2 under DSpark. The Qwen4-exp
/// and `Decoder` verifies read an argmax back inside the pass and draw
/// nothing, so a temperature is refused there until a sampled verify is
/// built for them.
const SAMPLED_VERIFY: [&str; 1] = ["dspark"];

/// C11 for a resolved speculative method.
pub fn refuse_speculative(method: &str, sampling: &crate::SamplingConfig) -> Result<(), String> {
    if !DRAFTERS.contains(&method) {
        return Ok(());
    }
    let mut why = Vec::new();
    let s = sampling.sampling();
    if host_side(&s) {
        why.push("a host-side sampler (generation.top_k / top_p: a host sort on one running rng, which a verify window cannot key by position)");
    } else if s != Sampling::ArgMax && !SAMPLED_VERIFY.contains(&method) {
        why.push("a temperature (generation.temperature: this target's verify reads an argmax back and draws nothing; a sampled verify is built for dspark only)");
    }
    if sampling.processors().is_active() {
        why.push("a logits processor (generation.min_p, the penalties, generation.logit_bias)");
    }
    if sampling.thinking_budget.is_some() {
        why.push("a thinking budget (reasoning.budget)");
    }
    if why.is_empty() {
        return Ok(());
    }
    Err(format!(
        "C11: speculative.method = {method} refuses {}",
        why.join(", ")
    ))
}

/// The static refusals, by id: what no arm can serve, refused when the
/// options resolve rather than at the first request.
pub fn refuse(o: &EngineOptions) -> Result<(), String> {
    let s = &o.sampling;
    if o.decode.readback.kind == crate::ReadbackKind::Ring && host_side(&s.sampling()) {
        return Err(format!(
            "C2: decode.readback.kind = ring with a host-side sampler ({:?}): the ring cannot draw it; \
             `auto` takes the blocking arm for it",
            s.sampling()
        ));
    }
    if o.decode.readback.lookahead > 1 {
        return Err(format!(
            "C10: decode.readback.lookahead_forwards = {} is not 0 or 1",
            o.decode.readback.lookahead
        ));
    }
    refuse_speculative(&o.speculative.method, s)?;
    if o.scheduler.max_sequences > 1 && (s.processors().is_active() || s.thinking_budget.is_some())
    {
        return Err(format!(
            "C12: scheduler.max_sequences = {} refuses {}: the batched driver's rows draw without \
             processors and feed no forced token (serve it at a batch of 1)",
            o.scheduler.max_sequences,
            if s.thinking_budget.is_some() {
                "a thinking budget (reasoning.budget)"
            } else {
                "a logits processor (generation.min_p, the penalties, generation.logit_bias)"
            }
        ));
    }
    Ok(())
}

/// What the couplings do to these options, one line each, in the key
/// vocabulary — `lloom config --explain`'s second half.
pub fn explain(o: &EngineOptions) -> Vec<String> {
    let mut out = Vec::new();
    let r = &o.decode.readback;
    let s = &o.sampling;
    let sampling = s.sampling();
    if r.kind == crate::ReadbackKind::Auto && host_side(&sampling) {
        let by = match (s.top_k.is_some(), s.top_p.is_some()) {
            (true, true) => "generation.top_k, generation.top_p",
            (true, false) => "generation.top_k",
            _ => "generation.top_p",
        };
        out.push(format!(
            "decode.readback.kind = auto -> blocking <- {by} (C1)"
        ));
    }
    if sampling == Sampling::ArgMax && (s.top_k.is_some() || s.top_p.is_some() || s.gumbel) {
        out.push("generation.top_k / top_p / device_sampler ignored <- generation.temperature unset (C3)".into());
    }
    if let Some(note) = r.adaptive_clamp() {
        out.push(format!(
            "decode.readback.adaptive_commit -> off <- {note} (C4, C5)"
        ));
    }
    if o.executor.replay {
        if r.eos_gate {
            out.push("decode.readback.eos_gate -> off <- executor.replay.enabled (C7)".into());
        }
        out.push(
            "executor.packed, executor.arena.enabled -> on <- executor.replay.enabled (C8)".into(),
        );
    }
    // The arm that runs: the blocking one where C1 resolves `auto` to it.
    let blocking = r.kind == crate::ReadbackKind::Blocking
        || (r.kind == crate::ReadbackKind::Auto && host_side(&sampling));
    let adaptive = !blocking && r.ring_arm().adaptive;
    if adaptive && !r.stream_reader {
        out.push(
            "decode.readback.stream_reader -> on <- decode.readback.adaptive_commit (C9)".into(),
        );
    }
    if r.eos != crate::EosKind::Poll && r.poll != lloom::generation::poll::PollMechanism::default()
    {
        out.push(format!(
            "decode.readback.wait ignored <- decode.readback.eos = {} (C13)",
            r.eos
        ));
    }
    if !matches!(
        r.eos,
        crate::EosKind::HostDrain
            | crate::EosKind::GpuBatched
            | crate::EosKind::GpuBatchedNonblocking
    ) && r.wake_cadence != lloom::generation::WAKE_LIVE_EVERY_DEFAULT
    {
        out.push(format!(
            "decode.readback.wake_every_commits ignored <- decode.readback.eos = {} (C14)",
            r.eos
        ));
    }
    if !adaptive && (r.run_ahead != 0 || !r.fused_publish) {
        out.push("decode.readback.run_ahead_steps / fused_publish ignored <- the adaptive commit does not run (C15)".into());
    }
    if DRAFTERS.contains(&o.speculative.method.as_str()) && o.model.cache == "auto" {
        out.push(format!(
            "models.lfm2.preset = auto -> rewindable <- speculative.method = {} (C17)",
            o.speculative.method
        ));
    }
    if o.scheduler.max_sequences > 1 && o.scheduler.rows == "auto" {
        out.push(format!(
            "scheduler.rows = auto -> pool <- scheduler.max_sequences = {} (C18)",
            o.scheduler.max_sequences
        ));
    }
    if o.executor.pass_plans && !o.executor.pass_plans_armed() {
        out.push(
            "executor.pass_plans -> off <- executor.arena.enabled / executor.replay.enabled (C19)"
                .into(),
        );
    }
    out
}

/// The schema's top-level `x-lloom-couplings`.
pub fn table() -> Value {
    Value::Array(
        COUPLINGS
            .iter()
            .map(|c| {
                json!({
                    "id": c.id, "kind": c.kind, "when": c.when, "effect": c.effect,
                    "keys": c.keys, "site": c.site, "static": c.is_static,
                })
            })
            .collect(),
    )
}

/// The static couplings as JSON Schema 2020-12 `if/then` rules over a config
/// file: what [`refuse`] refuses, checked before load by any validator.
pub fn static_rules() -> Value {
    let hot = json!({"type": "number", "minimum": 1e-7});
    let host_sampler = json!({"anyOf": [
        {"properties": {"top_k": {"type": "integer"}}, "required": ["top_k"]},
        {"properties": {"top_p": {"type": "number"}}, "required": ["top_p"]}
    ]});
    let active = json!({"anyOf": [
        {"properties": {"repetition_penalty": {"type": "number", "not": {"const": 1}}}, "required": ["repetition_penalty"]},
        {"properties": {"presence_penalty": {"type": "number", "not": {"const": 0}}}, "required": ["presence_penalty"]},
        {"properties": {"frequency_penalty": {"type": "number", "not": {"const": 0}}}, "required": ["frequency_penalty"]},
        {"properties": {"min_p": {"type": "number", "exclusiveMinimum": 0}}, "required": ["min_p"]},
        {"properties": {"logit_bias": {"type": "object", "minProperties": 1}}, "required": ["logit_bias"]}
    ]});
    let budget = json!({"properties": {"reasoning": {"properties": {"budget": {"type": "integer"}}, "required": ["budget"]}}, "required": ["reasoning"]});
    json!([
        {"$comment": "C2: an explicit ring with a host-side sampler is refused",
         "if": {"properties": {
                "decode": {"properties": {"readback": {"properties": {"kind": {"const": "ring"}}, "required": ["kind"]}}, "required": ["readback"]},
                "generation": {"allOf": [
                    {"properties": {"temperature": hot}, "required": ["temperature"]},
                    host_sampler,
                    {"not": {"properties": {"device_sampler": {"const": true}}, "required": ["device_sampler"]}}]}},
                "required": ["decode", "generation"]},
         "then": false},
        {"$comment": "C10: at most one forward past an unlearned EOS",
         "if": {"properties": {"decode": {"properties": {"readback": {"properties": {"lookahead_forwards": {"type": "integer", "minimum": 2}}, "required": ["lookahead_forwards"]}}, "required": ["readback"]}}, "required": ["decode"]},
         "then": false},
        {"$comment": "C11: a drafter refuses a host-side sampler, a logits processor and a thinking budget",
         "if": {"properties": {"speculative": {"properties": {"method": {"enum": DRAFTERS}}, "required": ["method"]}}, "required": ["speculative"]},
         "then": {"allOf": [
            {"not": {"properties": {"generation": {"allOf": [
                {"properties": {"temperature": hot}, "required": ["temperature"]},
                host_sampler,
                {"not": {"properties": {"device_sampler": {"const": true}}, "required": ["device_sampler"]}}]}}, "required": ["generation"]}},
            {"not": {"properties": {"generation": active}, "required": ["generation"]}},
            {"not": budget}]}},
        {"$comment": "C11: a drafter whose target's verify draws nothing (mtp, dflash) refuses a temperature",
         "if": {"properties": {"speculative": {"properties": {"method": {"enum": ["mtp", "dflash"]}}, "required": ["method"]}}, "required": ["speculative"]},
         "then": {"not": {"properties": {"generation": {"properties": {"temperature": hot}, "required": ["temperature"]}}, "required": ["generation"]}}},
        {"$comment": "C12: the batch scheduler refuses a logits processor and a thinking budget",
         "if": {"properties": {"scheduler": {"properties": {"max_sequences": {"type": "integer", "minimum": 2}}, "required": ["max_sequences"]}}, "required": ["scheduler"]},
         "then": {"allOf": [
            {"not": {"properties": {"generation": active}, "required": ["generation"]}},
            {"not": budget}]}}
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::option_tree::index;

    fn with(sets: &[(&str, &str)]) -> EngineOptions {
        let mut o = EngineOptions::default();
        for (k, v) in sets {
            index().set(&mut o, k, v).unwrap();
        }
        o
    }

    /// Each static coupling refuses by its id, and the shape next to it
    /// that the engine serves passes.
    #[test]
    fn each_static_coupling_refuses_by_name() {
        for (sets, id) in [
            (
                &[
                    ("decode.readback.kind", "ring"),
                    ("generation.temperature", "0.7"),
                    ("generation.top_p", "0.9"),
                ][..],
                "C2",
            ),
            (
                &[
                    ("speculative.method", "mtp"),
                    ("generation.temperature", "0.7"),
                ][..],
                "C11",
            ),
            (
                &[
                    ("speculative.method", "dspark"),
                    ("generation.min_p", "0.1"),
                ][..],
                "C11",
            ),
            (
                &[
                    ("speculative.method", "dspark"),
                    ("generation.temperature", "0.7"),
                    ("generation.top_p", "0.9"),
                ][..],
                "C11",
            ),
            (
                &[
                    ("speculative.method", "dflash"),
                    ("generation.temperature", "0.7"),
                ][..],
                "C11",
            ),
            (
                &[("speculative.method", "mtp"), ("reasoning.budget", "64")][..],
                "C11",
            ),
            (
                &[
                    ("scheduler.max_sequences", "2"),
                    ("generation.repetition_penalty", "1.1"),
                ][..],
                "C12",
            ),
            (
                &[("scheduler.max_sequences", "4"), ("reasoning.budget", "64")][..],
                "C12",
            ),
        ] {
            let err = refuse(&with(sets)).unwrap_err();
            assert!(err.starts_with(&format!("{id}:")), "{sets:?}: {err}");
        }
        for sets in [
            &[
                ("decode.readback.kind", "ring"),
                ("generation.temperature", "0.7"),
            ][..],
            &[
                ("decode.readback.kind", "auto"),
                ("generation.temperature", "0.7"),
                ("generation.top_p", "0.9"),
            ][..],
            &[
                ("decode.readback.kind", "ring"),
                ("generation.temperature", "0.7"),
                ("generation.top_p", "0.9"),
                ("generation.device_sampler", "true"),
            ][..],
            &[
                ("speculative.method", "auto"),
                ("generation.temperature", "0.7"),
            ][..],
            &[("speculative.method", "mtp"), ("generation.top_p", "0.9")][..],
            &[
                ("speculative.method", "dspark"),
                ("generation.temperature", "0.7"),
            ][..],
            &[
                ("speculative.method", "dspark"),
                ("generation.temperature", "0.7"),
                ("generation.device_sampler", "true"),
            ][..],
            &[
                ("speculative.method", "mtp"),
                ("generation.repetition_penalty", "1"),
            ][..],
            &[
                ("scheduler.max_sequences", "2"),
                ("generation.temperature", "0.7"),
            ][..],
            &[("scheduler.max_sequences", "1"), ("reasoning.budget", "64")][..],
        ] {
            refuse(&with(sets)).unwrap_or_else(|e| panic!("{sets:?}: {e}"));
        }
        let mut o = EngineOptions::default();
        o.decode.readback.lookahead = 2;
        assert!(refuse(&o).unwrap_err().starts_with("C10:"));
    }

    #[test]
    fn every_coupling_names_keys_of_the_tree() {
        let idx = index();
        for c in COUPLINGS {
            for k in c.keys {
                assert!(idx.keys.contains_key(*k), "{}: `{k}` is not a key", c.id);
            }
        }
    }

    #[test]
    fn explain_names_the_resolution_and_its_cause() {
        let lines = explain(&with(&[
            ("generation.temperature", "0.7"),
            ("generation.top_k", "40"),
        ]));
        assert!(
            lines
                .iter()
                .any(|l| l == "decode.readback.kind = auto -> blocking <- generation.top_k (C1)"),
            "{lines:?}"
        );
        assert_eq!(
            explain(&EngineOptions::default()),
            vec!["decode.readback.stream_reader -> on <- decode.readback.adaptive_commit (C9)"],
            "at the defaults only the reader the adaptive commit drains through"
        );
    }
}
