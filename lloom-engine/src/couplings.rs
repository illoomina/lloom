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
        keys: &["decode.readback.kind", "decode.readback.device_filter", "generation.top_k", "generation.top_p", "generation.temperature", "generation.sampling.method"],
        site: "lloom-engine/src/decode.rs DecodeParams::sampler_route; ReadbackArm::{ring_admits, resolve}",
        is_static: false,
    },
    Coupling {
        id: "C2",
        kind: "resolves",
        when: "decode.readback.kind = ring with the host-side sampler of C1",
        effect: "the request runs on the blocking arm — the ring has no host round trip for the draw — and the turn line and `timings.readback_note` say so (a warning, not a refusal, since 2026-10-20)",
        keys: &["decode.readback.kind", "decode.readback.device_filter", "generation.top_k", "generation.top_p", "generation.temperature", "generation.sampling.method"],
        site: "lloom-engine/src/decode.rs DecodeParams::sampler_route; ReadbackArm::{resolve, demotion}; the ring itself still refuses the pair (lloom/src/generation/readback.rs Ring::check_sampler)",
        is_static: false,
    },
    Coupling {
        id: "C3",
        kind: "ignored",
        when: "generation.temperature unset or below 1e-7",
        effect: "top_k, top_p and the sampling method have no effect: argmax runs",
        keys: &["generation.top_k", "generation.top_p", "generation.sampling.method", "generation.temperature"],
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
        when: "speculative.method is a drafter at scheduler.max_sequences > 1, and a temperature, a logits processor or a thinking budget is set",
        effect: "refused: the batched speculative drivers (the pipelined rows, the pool rows) verify by the argmax; a single sequence's speculative body serves every sampler, processor and budget",
        keys: &["speculative.method", "scheduler.max_sequences", "generation.temperature", "generation.min_p", "generation.repetition_penalty", "generation.presence_penalty", "generation.frequency_penalty", "generation.logit_bias", "reasoning.budget"],
        site: "lloom-engine/src/app/lfm2.rs static_dspark_body, lloom/src/nn/batched_spec.rs admit_with_eos; at resolve and after `auto` resolves: couplings::refuse",
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
        effect: "decode.readback.run_ahead_steps, decode.readback.queued_steps and decode.readback.fused_publish have no effect",
        keys: &["decode.readback.run_ahead_steps", "decode.readback.queued_steps", "decode.readback.fused_publish", "decode.readback.adaptive_commit"],
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
    Coupling {
        id: "C20",
        kind: "ignored",
        when: "kernels.gemm.f16_rows off, determinism = deterministic, scheduler.max_sequences = 1, the CPU backend, or a batched step with more live rows than kernels.gemm.f16_rows_max",
        effect: "kernels.gemm.f16_rows_decode has no effect: the step's matmuls take the GEMM (the GEMV at one row; the one-row reduction per row under deterministic). Under fast a step with prompt pieces puts its live rows on the decode route and its pieces on the GEMM, so a row's route, and so its bits, changes only where a decode step's live rows cross 1 and kernels.gemm.f16_rows_max: the step's two route seams. serve and bench default the arm on only at scheduler.max_sequences up to kernels.gemm.f16_rows_max (12 there on Metal, decision 60), where no step crosses the ceiling",
        keys: &["kernels.gemm.f16_rows_decode", "kernels.gemm.f16_rows", "kernels.gemm.f16_rows_max", "determinism", "scheduler.max_sequences", "backend.device"],
        site: "lloom-tensor/src/kernels/kernels/mlx_gemm.rs f16_rows_route, f16_rows_lead; lloom/src/nn/batched.rs BatchDecoder::forward_live (tuning::DecodeRows), BatchDecoder::forward_ragged_live (tuning::RaggedDecodeRows)",
        is_static: false,
    },
    Coupling {
        id: "C21",
        kind: "resolves",
        when: "kernels.residency.set_cap_pct in 1..=99, or kernels.residency.attach = heartbeat",
        effect: "the device's residency sets: one per class up to the cap, at most 24, the keep-warm requesting kernels.residency.hot; and whether they are on the Metal 3 queue (Metal 4 keeps the attachment). chat and serve detach them by default on Metal (decision 56); the library keeps them on the queue",
        keys: &["kernels.residency.set_cap_pct", "kernels.residency.attach", "kernels.residency.hot"],
        site: "lloom-tensor/src/kernels/metal/residency_set.rs ResidencyPolicy::selected, ResidencyAttachment::decide",
        is_static: false,
    },
    Coupling {
        id: "C22",
        kind: "ignored",
        when: "kernels.gemm.f16_rows off, determinism = deterministic, or the CPU backend",
        effect: "kernels.gemm.f16_rows_round has no effect: no launch takes the row axis. Where one does, a launch of an odd number of rows takes the next even capacity and every row keeps the one-row GEMV's bits, so the arm moves time, never a bit",
        keys: &["kernels.gemm.f16_rows_round", "kernels.gemm.f16_rows", "determinism", "backend.device"],
        site: "lloom-tensor/src/kernels/kernels/mlx_gemm.rs f16_rows_capacity",
        is_static: false,
    },
    Coupling {
        id: "C24",
        kind: "resolves",
        when: "kernels.residency.metal4 other than slots, or the model is Qwen4-exp on chat or serve",
        effect: "what a Metal 4 command buffer declares resident: the device's residency sets on the Metal 4 queue (queue-*) or used by each command buffer (cb-*), and under *-diff its slot set lists only what they do not cover. chat and serve take queue-diff for Qwen4-exp on Metal (decision 62), slots elsewhere; on Metal 3 there are no slot sets and the arm has no effect",
        keys: &["kernels.residency.metal4", "backend.metal.command_buffer.api", "model.path"],
        site: "lloom-tensor/src/kernels/metal/residency_set.rs Metal4Declare::selected; lloom-tensor/src/kernels/metal/metal4.rs Queue::new, Slot::new; lloom-engine/src/frontend.rs When::Qwen4Exp",
        is_static: false,
    },
    Coupling {
        id: "C25",
        kind: "resolves",
        when: "decode.batched.cut_hold other than off, or the model is LFM2 on serve or bench",
        effect: "whether the batched loop's decode-only forwards hold the counter cut off: ahead over a forward encoded ahead of the step's readback, all over every one, the blocking (sampled) arm too; a step that feeds prompt pieces keeps the period. serve and bench take all for LFM2 on Metal (decision 71), off elsewhere; at scheduler.max_sequences = 1 or on the CPU no batched loop cuts and the arm has no effect",
        keys: &["decode.batched.cut_hold", "scheduler.max_sequences", "model.path", "backend.device"],
        site: "lloom/src/nn/batched.rs BatchCutHold::{configured, hold}, BatchDecoder::step; lloom-engine/src/frontend.rs When::ModelAxes",
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
    !crate::ReadbackArm::ring_admits(s)
}

const DRAFTERS: [&str; 3] = ["dspark", "mtp", "dflash"];

/// C11 for a resolved speculative method at `max_sequences` rows: the
/// single-sequence speculative body draws each window row with the
/// request's sampler (`Probes::verify_and_probe_sampled`), so only the
/// batched speculative drivers, which verify by the argmax, refuse.
pub fn refuse_speculative(
    method: &str,
    sampling: &crate::SamplingConfig,
    max_sequences: usize,
) -> Result<(), String> {
    if !DRAFTERS.contains(&method) || max_sequences <= 1 {
        return Ok(());
    }
    let arm = sampling.sampling()?;
    let processors = sampling.processors().is_active();
    let refusal = match lloom::nn::refuse(
        lloom::nn::Body::BatchedWindow,
        &lloom::nn::Asks {
            sampling: &arm,
            processors,
            budget: sampling.thinking_budget.is_some(),
            digest: false,
            logprobs: None,
        },
    ) {
        Ok(()) => return Ok(()),
        Err(e) => e,
    };
    let mut why = Vec::new();
    if arm != Sampling::ArgMax {
        why.push("a temperature (generation.temperature)");
    }
    if processors {
        why.push("a logits processor (generation.min_p, the penalties, generation.logit_bias)");
    }
    if sampling.thinking_budget.is_some() {
        why.push("a thinking budget (reasoning.budget)");
    }
    if why.is_empty() {
        return Err(format!("C11: speculative.method = {method}: {refusal}"));
    }
    Err(format!(
        "C11: speculative.method = {method} at scheduler.max_sequences = {max_sequences} refuses \
         {}: the batched speculative drivers verify by the argmax (serve it at a batch of 1)",
        why.join(", ")
    ))
}

/// The static refusals, by id: what no arm can serve, refused when the
/// options resolve rather than at the first request.
pub fn refuse(o: &EngineOptions) -> Result<(), String> {
    let s = &o.sampling;
    let sampling = s.sampling()?;
    if o.decode.readback.lookahead > 1 {
        return Err(format!(
            "C10: decode.readback.lookahead_forwards = {} is not 0 or 1",
            o.decode.readback.lookahead
        ));
    }
    refuse_speculative(&o.speculative.method, s, o.scheduler.max_sequences)?;
    if o.scheduler.max_sequences > 1
        && lloom::nn::refuse(
            lloom::nn::Body::BatchedStep,
            &lloom::nn::Asks {
                sampling: &sampling,
                processors: s.processors().is_active(),
                budget: s.thinking_budget.is_some(),
                digest: false,
                logprobs: None,
            },
        )
        .is_err()
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
    let sampling = match s.sampling() {
        Ok(sampling) => sampling,
        Err(e) => {
            out.push(e);
            return out;
        }
    };
    // A load-time prediction for scalar defaults only. Actual request params,
    // model capability and memory admission still decide whether it can run.
    let scalar = o.scheduler.max_sequences <= 1 && o.speculative.method == "none";
    let route = if scalar {
        let params = crate::DecodeParams {
            readback: r.arm(),
            logits_digest: o.decode.logits_digest,
            ..Default::default()
        };
        Some(params.sampler_route(
            r.device_filter,
            &sampling,
            cfg!(feature = "metal") && o.load.device != "cpu",
            r.ring_arm(),
        ))
    } else {
        None
    };
    if r.device_filter {
        match &route {
            Some(Ok(selected)) if selected.prepared_sampler => out.push(format!(
                "decode.readback.device_filter -> {} (selected default-request route; model capability/admission still required)",
                selected.arm.name()
            )),
            Some(Err(error)) => out.push(format!(
                "decode.readback.device_filter: default request would refuse: {error}"
            )),
            _ => {}
        }
    }
    let host_demotion = match &route {
        Some(Ok(selected)) => !selected.prepared_sampler && !o.decode.logits_digest && host_side(&sampling),
        Some(Err(_)) => false,
        None => host_side(&sampling),
    };
    if host_demotion {
        let k = s.top_k.is_some_and(|k| k > 0);
        let p = s.top_p.is_some_and(|p| p < 1.0);
        let by = match (k, p) {
            (true, true) => "generation.top_k, generation.top_p",
            (true, false) => "generation.top_k",
            _ => "generation.top_p",
        };
        match r.kind {
            crate::ReadbackKind::Auto => out.push(format!(
                "decode.readback.kind = auto -> blocking <- {by} (C1)"
            )),
            crate::ReadbackKind::Ring => out.push(format!(
                "decode.readback.kind = ring -> blocking for this sampler <- {by}: the ring cannot \
                 draw a host sort; each request says so on its turn line (C2)"
            )),
            crate::ReadbackKind::Blocking => {}
        }
    }
    if sampling == Sampling::ArgMax && (s.top_k.is_some() || s.top_p.is_some() || s.gumbel) {
        out.push("generation.top_k / top_p / sampling.method ignored <- generation.temperature unset (C3)".into());
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
    let blocking = r.kind == crate::ReadbackKind::Blocking || host_side(&sampling);
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
    if !adaptive && (r.run_ahead != 0 || r.queued_steps != 0 || !r.fused_publish) {
        out.push("decode.readback.run_ahead_steps / queued_steps / fused_publish ignored <- the adaptive commit does not run (C15)".into());
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
    if let Some(line) = residency_layout(&o.tuning.metal) {
        out.push(line);
    }
    if let Some(line) = metal4_declaration(o) {
        out.push(line);
    }
    if let Some(line) = batched_cut_hold(o) {
        out.push(line);
    }
    let g = &o.tuning.gemm;
    let rows_cause = if o.tuning.contract.profile == "deterministic" {
        Some("determinism = deterministic")
    } else if !g.f16_rows {
        Some("kernels.gemm.f16_rows = false")
    } else if !cfg!(feature = "metal") || o.load.device == "cpu" {
        Some("the CPU backend")
    } else if o.scheduler.max_sequences <= 1 {
        Some("scheduler.max_sequences = 1")
    } else {
        None
    };
    let default_batch = format!(
        "serve and bench turn it on by default at a batch of 2 to {} sequences",
        g.f16_rows_max
    );
    if g.f16_rows_decode {
        out.push(match rows_cause {
            Some(c) => format!("kernels.gemm.f16_rows_decode ignored <- {c} (C20)"),
            None if o.scheduler.max_sequences > g.f16_rows_max => format!(
                "kernels.gemm.f16_rows_decode: a decode step of 2 to {} live rows (1 to {} beside \
                 prompt pieces) takes the row axis, of more the GEMM; a batch of {} sequences \
                 crosses the ceiling, so a row's route follows how many rows its steps hold, \
                 which follows the requests' arrival; {default_batch} <- \
                 kernels.gemm.f16_rows_max (C20)",
                g.f16_rows_max, g.f16_rows_max, o.scheduler.max_sequences
            ),
            None => format!(
                "kernels.gemm.f16_rows_decode: a decode step of 2 to {} live rows (1 to {} beside \
                 prompt pieces) takes the row axis, of more the GEMM; {default_batch} <- \
                 kernels.gemm.f16_rows_max (C20)",
                g.f16_rows_max, g.f16_rows_max
            ),
        });
    } else if rows_cause.is_none() && o.scheduler.max_sequences > g.f16_rows_max {
        out.push(format!(
            "kernels.gemm.f16_rows_decode off: {default_batch}, and a batch of {} is past the \
             ceiling, where a row's route would follow the requests' arrival <- \
             kernels.gemm.f16_rows_max (C20)",
            o.scheduler.max_sequences
        ));
    }
    if g.f16_rows_round {
        let cause = if o.tuning.contract.profile == "deterministic" {
            Some("determinism = deterministic")
        } else if !g.f16_rows {
            Some("kernels.gemm.f16_rows = false")
        } else if !cfg!(feature = "metal") || o.load.device == "cpu" {
            Some("the CPU backend")
        } else {
            None
        };
        out.push(match cause {
            Some(c) => format!("kernels.gemm.f16_rows_round ignored <- {c} (C22)"),
            None => "kernels.gemm.f16_rows_round: a row-axis launch of an odd number of rows \
                     takes the next even capacity, the same bits <- kernels.gemm.f16_rows (C22)"
                .into(),
        });
    }
    out
}

/// The residency router's layout as the options resolve it (`tuning.md`
/// §6.3n, §6.3o); `None` at the library's values (one set, attached to the
/// queue).
fn residency_layout(m: &lloom_tensor::tuning::MetalTuning) -> Option<String> {
    let pct = m.residency_set_cap_pct;
    let split = pct > 0 && pct < 100;
    if !split && m.residency_attach == "queue" {
        return None;
    }
    let sets = if split {
        format!(
            "sets per class (weights, kv, heap, scratch, experts, other), each at most {pct} % of \
             the device's recommended working set, at most {} sets; the keep-warm requests {}",
            lloom_tensor::residency_class::MAX_DEVICE_SETS,
            m.residency_hot
        )
    } else {
        "1 set".into()
    };
    let attach = if m.residency_attach == "heartbeat" {
        "detached from the Metal 3 queue (chat's and serve's default on Metal, decision 56; \
         Metal 4: attached as before)"
    } else {
        "on the Metal 3 queue"
    };
    Some(format!(
        "kernels.residency -> {sets}, {attach} <- kernels.residency.set_cap_pct = {pct}, \
         kernels.residency.attach = {} (C21)",
        m.residency_attach
    ))
}

/// What a Metal 4 command buffer declares resident as the options resolve it
/// (`tuning.md` §6.3p); `None` at `slots`, the library's value.
fn metal4_declaration(o: &EngineOptions) -> Option<String> {
    let arm = o.tuning.metal.residency_metal4.as_str();
    if arm == "slots" {
        return None;
    }
    if o.tuning.metal.command_api == "metal3" {
        return Some(format!(
            "kernels.residency.metal4 = {arm} ignored <- backend.metal.command_buffer.api = \
             metal3: no slot sets (C24)"
        ));
    }
    let sets = if arm.starts_with("queue") {
        "the device's residency sets on the Metal 4 queue"
    } else {
        "the device's residency sets used by each command buffer"
    };
    let lists = if arm.ends_with("diff") {
        "each slot set listing only what they do not cover"
    } else {
        "each slot set listing every binding (a control)"
    };
    let default = if arm == "queue-diff" && crate::frontend::When::Qwen4Exp.holds(o) {
        " (chat's and serve's default for Qwen4-exp on Metal, decision 62)"
    } else {
        ""
    };
    Some(format!(
        "kernels.residency.metal4 -> {sets}, {lists}{default} <- kernels.residency.metal4 = \
         {arm} (C24)"
    ))
}

/// The batched loop's counter cut as the options resolve it (`tuning.md`
/// §13.5r); `None` at `off`, the library's value.
fn batched_cut_hold(o: &EngineOptions) -> Option<String> {
    let arm = o.tuning.ring.batched_cut_hold.as_str();
    if arm == "off" {
        return None;
    }
    if !cfg!(feature = "metal") || o.load.device == "cpu" {
        return Some(format!(
            "decode.batched.cut_hold = {arm} ignored <- the CPU backend (C25)"
        ));
    }
    if o.scheduler.max_sequences <= 1 {
        return Some(format!(
            "decode.batched.cut_hold = {arm} ignored <- scheduler.max_sequences = 1: no batched \
             loop runs (C25)"
        ));
    }
    let held = if arm == "ahead" {
        "a decode-only forward encoded ahead of the step's readback"
    } else {
        "every decode-only forward, the blocking arm too"
    };
    let default = if arm == "all" && crate::frontend::When::ModelAxes.holds(o) {
        " (serve's and bench's default for LFM2 on Metal, decision 71)"
    } else {
        ""
    };
    Some(format!(
        "decode.batched.cut_hold -> the batched loop holds the counter cut off over {held}; a \
         step that feeds prompt pieces keeps the period{default} <- decode.batched.cut_hold = \
         {arm} (C25)"
    ))
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
    let active = json!({"anyOf": [
        {"properties": {"repetition_penalty": {"type": "number", "not": {"const": 1}}}, "required": ["repetition_penalty"]},
        {"properties": {"presence_penalty": {"type": "number", "not": {"const": 0}}}, "required": ["presence_penalty"]},
        {"properties": {"frequency_penalty": {"type": "number", "not": {"const": 0}}}, "required": ["frequency_penalty"]},
        {"properties": {"min_p": {"type": "number", "exclusiveMinimum": 0}}, "required": ["min_p"]},
        {"properties": {"logit_bias": {"type": "object", "minProperties": 1}}, "required": ["logit_bias"]}
    ]});
    let budget = json!({"properties": {"reasoning": {"properties": {"budget": {"type": "integer"}}, "required": ["budget"]}}, "required": ["reasoning"]});
    json!([
        {"$comment": "C10: at most one forward past an unlearned EOS",
         "if": {"properties": {"decode": {"properties": {"readback": {"type": "object", "properties": {"lookahead_forwards": {"type": "integer", "minimum": 2}}, "required": ["lookahead_forwards"]}}, "required": ["readback"]}}, "required": ["decode"]},
         "then": false},
        {"$comment": "C11: a drafter at more than one sequence refuses a temperature, a logits processor and a thinking budget",
         "if": {"properties": {
                "speculative": {"properties": {"method": {"enum": DRAFTERS}}, "required": ["method"]},
                "scheduler": {"properties": {"max_sequences": {"type": "integer", "minimum": 2}}, "required": ["max_sequences"]}},
                "required": ["speculative", "scheduler"]},
         "then": {"allOf": [
            {"not": {"properties": {"generation": {"properties": {"temperature": hot}, "required": ["temperature"]}}, "required": ["generation"]}},
            {"not": {"properties": {"generation": active}, "required": ["generation"]}},
            {"not": budget}]}},
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
                    ("speculative.method", "mtp"),
                    ("scheduler.max_sequences", "2"),
                    ("generation.temperature", "0.7"),
                ][..],
                "C11",
            ),
            (
                &[
                    ("speculative.method", "dspark"),
                    ("scheduler.max_sequences", "4"),
                    ("generation.min_p", "0.1"),
                ][..],
                "C11",
            ),
            (
                &[
                    ("speculative.method", "dflash"),
                    ("scheduler.max_sequences", "2"),
                    ("reasoning.budget", "64"),
                ][..],
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
            // C2 resolves (the blocking arm with a note) instead of refusing.
            &[
                ("decode.readback.kind", "ring"),
                ("generation.temperature", "0.7"),
                ("generation.top_p", "0.9"),
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
                ("generation.sampling.method", "gumbel"),
            ][..],
            // The single-sequence speculative body serves every sampler arm,
            // the processors and the budget, on every drafter.
            &[
                ("speculative.method", "mtp"),
                ("generation.temperature", "0.7"),
            ][..],
            &[
                ("speculative.method", "dflash"),
                ("generation.temperature", "0.7"),
                ("generation.top_k", "40"),
                ("generation.top_p", "0.9"),
            ][..],
            &[
                ("speculative.method", "dspark"),
                ("generation.min_p", "0.1"),
                ("generation.repetition_penalty", "1.1"),
            ][..],
            &[("speculative.method", "mtp"), ("reasoning.budget", "64")][..],
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
        let filtered_gumbel = with(&[
            ("decode.readback.kind", "ring"),
            ("generation.temperature", "0.7"),
            ("generation.top_p", "0.9"),
            ("generation.sampling.method", "gumbel"),
        ]);
        // A ring asked by name with filtered gumbel resolves to the blocking
        // arm (C2), as top-k / top-p does.
        refuse(&filtered_gumbel).unwrap();
        assert!(
            explain(&filtered_gumbel).iter().any(|l| l.starts_with(
                "decode.readback.kind = ring -> blocking for this sampler <- generation.top_p"
            ) && l.ends_with("(C2)")),
            "{:?}",
            explain(&filtered_gumbel)
        );
        let mut o = EngineOptions::default();
        o.decode.readback.lookahead = 2;
        assert!(refuse(&o).unwrap_err().starts_with("C10:"));
    }

    #[test]
    fn disabled_filters_stay_on_the_ring_but_filtered_gumbel_is_explicit_host() {
        for method in ["multinomial", "gumbel"] {
            let off = with(&[
                ("decode.readback.kind", "ring"),
                ("generation.temperature", "0.7"),
                ("generation.top_k", "0"),
                ("generation.top_p", "1"),
                ("generation.sampling.method", method),
            ]);
            refuse(&off).unwrap();
        }
        let host = with(&[
            ("decode.readback.kind", "auto"),
            ("generation.temperature", "0.7"),
            ("generation.top_p", "0"),
            ("generation.sampling.method", "gumbel"),
        ]);
        refuse(&host).unwrap();
        assert!(host.sampling.sampling().unwrap().requires_host());
        assert!(explain(&host)
            .iter()
            .any(|s| s.contains("auto -> blocking")));
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
        // A ring asked by name with a host-side sampler is explained (C2),
        // not refused, and the stream-reader line (C9) is not added for it:
        // the blocking arm runs.
        let lines = explain(&with(&[
            ("decode.readback.kind", "ring"),
            ("generation.temperature", "0.7"),
            ("generation.top_p", "0.9"),
        ]));
        assert!(
            lines.iter().any(|l| l.starts_with(
                "decode.readback.kind = ring -> blocking for this sampler <- generation.top_p"
            ) && l.ends_with("(C2)")),
            "{lines:?}"
        );
        assert!(!lines.iter().any(|l| l.contains("(C9)")), "{lines:?}");
        assert_eq!(
            explain(&EngineOptions::default()),
            vec!["decode.readback.stream_reader -> on <- decode.readback.adaptive_commit (C9)"],
            "at the defaults only the reader the adaptive commit drains through"
        );
        // C20: the decode rows' arm names its ceiling where it can take the
        // row axis, and its cause where it cannot.
        let c20 = |sets: &[(&str, &str)]| -> Vec<String> {
            explain(&with(sets))
                .into_iter()
                .filter(|l| l.ends_with("(C20)"))
                .collect()
        };
        let on = [
            ("kernels.gemm.f16_rows_decode", "true"),
            ("scheduler.max_sequences", "4"),
        ];
        let at_batch = c20(&on);
        if cfg!(feature = "metal") {
            assert_eq!(
                at_batch,
                vec![
                    "kernels.gemm.f16_rows_decode: a decode step of 2 to 10 live rows (1 to 10 \
                      beside prompt pieces) takes the row axis, of more the GEMM; serve and bench \
                      turn it on by default at a batch of 2 to 10 sequences <- \
                      kernels.gemm.f16_rows_max (C20)"
                ]
            );
            // Decision 60's ceiling, as serve and bench resolve it.
            assert_eq!(
                c20(&[
                    ("kernels.gemm.f16_rows_decode", "true"),
                    ("kernels.gemm.f16_rows_max", "12"),
                    ("scheduler.max_sequences", "12"),
                ]),
                vec![
                    "kernels.gemm.f16_rows_decode: a decode step of 2 to 12 live rows (1 to 12 \
                      beside prompt pieces) takes the row axis, of more the GEMM; serve and bench \
                      turn it on by default at a batch of 2 to 12 sequences <- \
                      kernels.gemm.f16_rows_max (C20)"
                ]
            );
        } else {
            assert_eq!(
                at_batch,
                vec!["kernels.gemm.f16_rows_decode ignored <- the CPU backend (C20)"]
            );
        }
        let mut wide = on.to_vec();
        wide[1] = ("scheduler.max_sequences", "16");
        if cfg!(feature = "metal") {
            assert_eq!(
                c20(&wide),
                vec![
                    "kernels.gemm.f16_rows_decode: a decode step of 2 to 10 live rows (1 to 10 \
                      beside prompt pieces) takes the row axis, of more the GEMM; a batch of 16 \
                      sequences crosses the ceiling, so a row's route follows how many rows its \
                      steps hold, which follows the requests' arrival; serve and bench turn it on \
                      by default at a batch of 2 to 10 sequences <- kernels.gemm.f16_rows_max (C20)"
                ]
            );
        }
        let mut one = on.to_vec();
        one[1] = ("scheduler.max_sequences", "1");
        if cfg!(feature = "metal") {
            assert_eq!(
                c20(&one),
                vec!["kernels.gemm.f16_rows_decode ignored <- scheduler.max_sequences = 1 (C20)"]
            );
        }
        let mut det = on.to_vec();
        det.push(("determinism", "deterministic"));
        assert_eq!(
            c20(&det),
            vec!["kernels.gemm.f16_rows_decode ignored <- determinism = deterministic (C20)"]
        );
        let mut rows_off = on.to_vec();
        rows_off.push(("kernels.gemm.f16_rows", "false"));
        assert_eq!(
            c20(&rows_off),
            vec!["kernels.gemm.f16_rows_decode ignored <- kernels.gemm.f16_rows = false (C20)"]
        );
        assert!(
            c20(&[("scheduler.max_sequences", "4")]).is_empty(),
            "the arm off"
        );
        // Off past the ceiling: the default batch it would take, named.
        let past = c20(&[
            ("kernels.gemm.f16_rows_max", "12"),
            ("scheduler.max_sequences", "13"),
        ]);
        if cfg!(feature = "metal") {
            assert_eq!(
                past,
                vec![
                    "kernels.gemm.f16_rows_decode off: serve and bench turn it on by default at a \
                      batch of 2 to 12 sequences, and a batch of 13 is past the ceiling, where a \
                      row's route would follow the requests' arrival <- kernels.gemm.f16_rows_max \
                      (C20)"
                ]
            );
        } else {
            assert!(past.is_empty(), "{past:?}");
        }
        // C22: the capacity rounding names what it does where the row axis
        // runs, and its cause where it does not.
        let c22 = |sets: &[(&str, &str)]| -> Vec<String> {
            explain(&with(sets))
                .into_iter()
                .filter(|l| l.ends_with("(C22)"))
                .collect()
        };
        let round = [("kernels.gemm.f16_rows_round", "true")];
        if cfg!(feature = "metal") {
            assert_eq!(
                c22(&round),
                vec![
                    "kernels.gemm.f16_rows_round: a row-axis launch of an odd number of rows \
                     takes the next even capacity, the same bits <- kernels.gemm.f16_rows (C22)"
                ]
            );
        } else {
            assert_eq!(
                c22(&round),
                vec!["kernels.gemm.f16_rows_round ignored <- the CPU backend (C22)"]
            );
        }
        let mut det = round.to_vec();
        det.push(("determinism", "deterministic"));
        assert_eq!(
            c22(&det),
            vec!["kernels.gemm.f16_rows_round ignored <- determinism = deterministic (C22)"]
        );
        let mut rows_off = round.to_vec();
        rows_off.push(("kernels.gemm.f16_rows", "false"));
        assert_eq!(
            c22(&rows_off),
            vec!["kernels.gemm.f16_rows_round ignored <- kernels.gemm.f16_rows = false (C22)"]
        );
        assert!(c22(&[]).is_empty(), "the arm off");
    }

    /// C21 at the library's values says nothing; where chat's and serve's
    /// rows hold (Metal) their default detaches the set, and the line names
    /// decision 56; `queue` named is the library's arrangement again.
    #[test]
    fn c21_names_the_default_attachment() {
        use crate::frontend::{self, Frontend, When};
        let c21 = |o: &EngineOptions| -> Vec<String> {
            explain(o)
                .into_iter()
                .filter(|l| l.ends_with("(C21)"))
                .collect()
        };
        assert!(c21(&EngineOptions::default()).is_empty(), "the library's");
        for f in [Frontend::Chat, Frontend::Serve, Frontend::Bench] {
            let mut o = EngineOptions::default();
            frontend::apply(&mut o, f, &Default::default(), &Default::default()).unwrap();
            let detached = f != Frontend::Bench && When::Metal.holds(&o);
            let want: Vec<String> = if detached {
                vec!["kernels.residency -> 1 set, detached from the Metal 3 queue (chat's and \
                      serve's default on Metal, decision 56; Metal 4: attached as before) <- \
                      kernels.residency.set_cap_pct = 0, kernels.residency.attach = heartbeat (C21)"
                    .into()]
            } else {
                vec![]
            };
            assert_eq!(c21(&o), want, "{}", f.name());
        }
        assert!(c21(&with(&[("kernels.residency.attach", "queue")])).is_empty());
    }

    /// C24 at `slots` says nothing; on a Qwen4-exp checkpoint chat's and
    /// serve's default names decision 62; on LFM2 neither prints; the arm on
    /// Metal 3 is named as ignored, and a control is named as one.
    #[test]
    fn c24_names_the_metal4_declaration() {
        use crate::frontend::{self, Frontend, When};
        let c24 = |o: &EngineOptions| -> Vec<String> {
            explain(o)
                .into_iter()
                .filter(|l| l.ends_with("(C24)"))
                .collect()
        };
        assert!(c24(&EngineOptions::default()).is_empty(), "the library's");
        let dir = std::env::temp_dir().join(format!("lloom-c24-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let qwen = dir.join("qwen4exp.gguf");
        std::fs::write(&qwen, "").unwrap();
        for f in [Frontend::Chat, Frontend::Serve, Frontend::Bench] {
            for model in [None, Some(qwen.clone())] {
                let mut o = EngineOptions::default();
                o.load.model = model.clone();
                frontend::apply(&mut o, f, &Default::default(), &Default::default()).unwrap();
                let on = f != Frontend::Bench && model.is_some() && When::Qwen4Exp.holds(&o);
                let want: Vec<String> = if on {
                    vec![
                        "kernels.residency.metal4 -> the device's residency sets on the Metal 4 \
                          queue, each slot set listing only what they do not cover (chat's and \
                          serve's default for Qwen4-exp on Metal, decision 62) <- \
                          kernels.residency.metal4 = queue-diff (C24)"
                            .into(),
                    ]
                } else {
                    vec![]
                };
                assert_eq!(c24(&o), want, "{} {model:?}", f.name());
            }
        }
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(
            c24(&with(&[("kernels.residency.metal4", "cb-full")])),
            vec![
                "kernels.residency.metal4 = cb-full ignored <- backend.metal.command_buffer.api \
                  = metal3: no slot sets (C24)"
            ]
        );
        assert_eq!(
            c24(&with(&[
                ("kernels.residency.metal4", "cb-full"),
                ("backend.metal.command_buffer.api", "metal4")
            ])),
            vec![
                "kernels.residency.metal4 -> the device's residency sets used by each command \
                  buffer, each slot set listing every binding (a control) <- \
                  kernels.residency.metal4 = cb-full (C24)"
            ]
        );
    }
    /// C25 at `off` says nothing; serve's and bench's default on LFM2 names
    /// decision 71 at a batch and is named inert at one sequence; Qwen4-exp
    /// and chat keep `off`; `ahead` is named as itself, and the CPU as the
    /// reason the arm does nothing.
    #[test]
    fn c25_names_the_batched_cut_hold() {
        use crate::frontend::{self, Frontend, When};
        let c25 = |o: &EngineOptions| -> Vec<String> {
            explain(o)
                .into_iter()
                .filter(|l| l.ends_with("(C25)"))
                .collect()
        };
        assert!(c25(&EngineOptions::default()).is_empty(), "the library's");
        let dir = std::env::temp_dir().join(format!("lloom-c25-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let qwen = dir.join("qwen4exp.gguf");
        std::fs::write(&qwen, "").unwrap();
        for f in Frontend::ALL {
            for model in [None, Some(qwen.clone())] {
                for batch in [1usize, 4] {
                    let mut o = EngineOptions::default();
                    o.load.model = model.clone();
                    o.scheduler.max_sequences = batch;
                    frontend::apply(&mut o, f, &Default::default(), &Default::default()).unwrap();
                    let on = f != Frontend::Chat && model.is_none() && When::ModelAxes.holds(&o);
                    let want: Vec<String> = match (on, batch) {
                        (false, _) => vec![],
                        (true, 1) => vec!["decode.batched.cut_hold = all ignored <- \
                                           scheduler.max_sequences = 1: no batched loop runs (C25)"
                            .into()],
                        (true, _) => vec!["decode.batched.cut_hold -> the batched loop holds the \
                                           counter cut off over every decode-only forward, the \
                                           blocking arm too; a step that feeds prompt pieces keeps \
                                           the period (serve's and bench's default for LFM2 on \
                                           Metal, decision 71) <- decode.batched.cut_hold = all (C25)"
                            .into()],
                    };
                    assert_eq!(c25(&o), want, "{} {model:?} at {batch}", f.name());
                }
            }
        }
        std::fs::remove_dir_all(&dir).unwrap();
        let ahead = with(&[
            ("decode.batched.cut_hold", "ahead"),
            ("scheduler.max_sequences", "4"),
        ]);
        let want = if cfg!(feature = "metal") {
            "decode.batched.cut_hold -> the batched loop holds the counter cut off over a \
             decode-only forward encoded ahead of the step's readback; a step that feeds prompt \
             pieces keeps the period <- decode.batched.cut_hold = ahead (C25)"
        } else {
            "decode.batched.cut_hold = ahead ignored <- the CPU backend (C25)"
        };
        assert_eq!(c25(&ahead), vec![want]);
        assert_eq!(
            c25(&with(&[
                ("decode.batched.cut_hold", "all"),
                ("scheduler.max_sequences", "4"),
                ("backend.device", "cpu"),
            ])),
            vec!["decode.batched.cut_hold = all ignored <- the CPU backend (C25)"]
        );
    }
}
