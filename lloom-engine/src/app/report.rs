//! The run reports: what one decode run engaged and what it cost, read from
//! the loop's counters (`DecodeStats`) and from the device's — the buffer
//! pool, the residency set, the submission locks, the dispatch recorder, the
//! executors — sampled around the run. `lloom chat` prints them after a turn:
//! a report some flag or variable asked for always (`--replay`, `--arena`,
//! `--packed`, `--pass-plans`, a speculative body, `CANDLE_METAL_PROFILE`,
//! `CANDLE_METAL_TRACE`, `--mem-sample`), the diagnostics under `--verbose`.
//!
//! The line spellings are the ones the measurement corpus greps, so a run
//! through `lloom` reads like one through the harness these came from
//! (`decode-driver-design.md` Stage 7). A mechanism whose engagement nobody
//! counts is one whose null reads as a result (`DESIGN.md` §2.4), which is
//! why the executor lines print whenever their arm is installed.

use super::common::{MemGauge, MemReport};
use lloom::nn::{DecodeStats, StepPhases};
use lloom_tensor::Device;

/// What the installed executor arms are, kept for their engagement reports:
/// the ICB executor itself (its coverage lives on the instance), and which of
/// the arena, the packed parameters and the pass-plan cache were armed.
#[derive(Clone, Default)]
pub struct Installed {
    #[cfg(feature = "metal")]
    pub replay: Option<std::sync::Arc<lloom_tensor::kernels::IcbExecutor>>,
    pub arena: bool,
    pub packed: bool,
    pub pass_plans: bool,
    /// The `--pass-widths` arm, for the pass-plan line.
    pub pass_widths: &'static str,
}

impl Installed {
    fn any(&self) -> bool {
        self.arena || self.packed || self.pass_plans || self.replay_installed()
    }

    fn replay_installed(&self) -> bool {
        #[cfg(feature = "metal")]
        return self.replay.is_some();
        #[cfg(not(feature = "metal"))]
        false
    }
}

/// Which reports a turn prints beyond the ones its flags asked for, and the
/// knobs their lines name.
#[derive(Clone, Copy, Debug, Default)]
pub struct Wanted {
    /// The once-per-run diagnostics (`--verbose`).
    pub verbose: bool,
    /// `decode.phase_sync`: the speculative round's phases were drained.
    pub phase_sync: bool,
    /// `decode.readback.run_ahead` (0 is the loop's 8).
    pub run_ahead: usize,
}

/// What the run was, for the lines that divide by it.
#[derive(Clone, Copy, Debug)]
pub struct RunFacts {
    /// Tokens the run produced.
    pub tokens: usize,
    pub prompt_tokens: usize,
    /// The run's own bracket, prefill included.
    pub wall: std::time::Duration,
    /// The speculative block, when a speculative body ran.
    pub k: Option<usize>,
}

/// The device's counters at the start of a run, so the run's own churn,
/// contention and page faults print as deltas; the profiler is reset so its
/// window is the run's.
pub struct RunProbe {
    #[cfg(feature = "metal")]
    churn: Option<ChurnSnapshot>,
    #[cfg(feature = "metal")]
    contention: Option<lloom_tensor::kernels::metal::contention::SubmissionContention>,
    #[cfg(feature = "metal")]
    pool_contention: Option<lloom_tensor::kernels::metal::contention::PoolContention>,
    faults: (u64, u64),
    mem: Option<MemGauge>,
    /// The run's start, for the profile's window (Metal only).
    #[cfg_attr(not(feature = "metal"), allow(dead_code))]
    started: std::time::Instant,
}

impl RunProbe {
    /// Snapshot the counters; `mem_sample` starts the 10 ms memory sampler.
    pub fn start(device: &Device, mem_sample: bool) -> Self {
        #[cfg(feature = "metal")]
        if lloom_tensor::kernels::metal::profile::enabled() {
            lloom_tensor::kernels::metal::profile::reset();
        }
        RunProbe {
            #[cfg(feature = "metal")]
            churn: match device {
                Device::Metal(m) => Some(churn_snapshot(m)),
                _ => None,
            },
            #[cfg(feature = "metal")]
            contention: match device {
                Device::Metal(m) => Some(m.commands().submission_contention()),
                _ => None,
            },
            #[cfg(feature = "metal")]
            pool_contention: match device {
                Device::Metal(m) => Some(m.pool_contention()),
                _ => None,
            },
            faults: rusage_faults(),
            mem: mem_sample.then(|| MemGauge::start(device, true)),
            started: std::time::Instant::now(),
        }
    }

    /// Print the run's reports.
    pub fn finish(
        self,
        device: &Device,
        stats: &DecodeStats,
        facts: RunFacts,
        installed: Option<&Installed>,
        wanted: Wanted,
    ) {
        let RunFacts {
            tokens, wall, k, ..
        } = facts;
        let faults = rusage_faults();
        let mem = self.mem.map(|g| g.finish(device));
        if wanted.verbose {
            print_diagnostics(stats, (self.faults, faults), facts, wanted);
            #[cfg(feature = "metal")]
            if let Device::Metal(m) = device {
                if let Some(before) = &self.contention {
                    let d = m.commands().submission_contention().since(before);
                    eprintln!(
                        "[lock contention over the run: waits={} host_blocked={:.1} us over {} steps | {}]",
                        d.total_waits(),
                        d.host_blocked_ns() as f64 / 1e3,
                        stats.steps,
                        d.render(),
                    );
                }
                if let Some(before) = &self.pool_contention {
                    let d = m.pool_contention().since(before);
                    eprintln!(
                        "[pool lock contention over the run: waits={} host_blocked={:.1} us over {} steps | {}]",
                        d.total_waits(),
                        d.host_blocked_ns() as f64 / 1e3,
                        stats.steps,
                        d.render(),
                    );
                }
                if let Some(before) = &self.churn {
                    print_churn(m, before);
                }
            }
        }
        print_census_and_phases(stats);
        if let Some(g) = stats.eos_gate {
            eprintln!(
                "[eos gate: fired={} gate kernels={}; overrun {} forwards{}]",
                g.fired,
                g.steps,
                stats.overrun,
                if stats.overrun_unwritten {
                    ", none of which wrote the cache (reconciled by counters)"
                } else {
                    ""
                },
            );
        }
        if let Some(k) = k {
            print_speculative(stats, tokens, wall, k, wanted);
        }
        if let Some(m) = &mem {
            print_mempeak(m);
        }
        if let Some(i) = installed {
            print_executor(device, i, stats.steps);
        }
        // The profile's window closes when the device drains: a turn can
        // return with its last lookahead forward still running, and a busy
        // union read against the call's wall alone would exceed it.
        #[cfg(feature = "metal")]
        if lloom_tensor::kernels::metal::profile::enabled() && device.synchronize().is_ok() {
            print_profile(stats, self.started.elapsed(), wall);
        }
        #[cfg(feature = "metal")]
        print_trace_census(stats.steps);
        print_step_shape(stats, wanted.verbose);
    }
}

/// The once-per-run diagnostics: the hot thread's scheduling class and the
/// faults taken, the adaptive commit's closes, the first steps' split.
fn print_diagnostics(
    stats: &DecodeStats,
    faults: ((u64, u64), (u64, u64)),
    facts: RunFacts,
    wanted: Wanted,
) {
    let h = &stats.hot_thread;
    eprintln!(
        "[hot thread: qos {} → {} (arm {:?}{}); page faults during decode: minor {}, major {}]",
        h.before.as_str(),
        h.after.as_str(),
        h.arm,
        if h.engaged() {
            ", engaged"
        } else if h.set_errno != 0 {
            ", set refused"
        } else {
            ""
        },
        faults.1 .0.saturating_sub(faults.0 .0),
        faults.1 .1.saturating_sub(faults.0 .1),
    );
    if stats.adaptive_commits + stats.run_ahead_closes > 0 {
        eprintln!(
            "[adaptive commit: {} buffers closed on the GPU starting the previous one + {} at the \
             run-ahead bound ({}), over {} steps = {:.1} steps/buffer, max {} in one; stopped by \
             the reader's EOS flag: {}; fused publishes {}/{}]",
            stats.adaptive_commits,
            stats.run_ahead_closes,
            if wanted.run_ahead == 0 {
                8
            } else {
                wanted.run_ahead
            },
            stats.steps,
            stats.steps as f64 / (stats.adaptive_commits + stats.run_ahead_closes).max(1) as f64,
            stats.max_steps_per_buffer,
            stats.stopped_by_flag,
            stats.readback.fused_publishes,
            stats.readback.ring_publishes,
        );
    }
    if !stats.step_ns.is_empty() {
        let ms = |ns: u64| ns as f64 / 1e6;
        eprintln!(
            "[ttft split: prefill {:.2} ms + step0 {:.2} ms; step_ns[1..={}] = {}; prompt_tokens={}]",
            ms(stats.prefill_ns),
            ms(stats.step_ns.first().copied().unwrap_or(0)),
            stats.step_ns.len().min(5).saturating_sub(1),
            stats
                .step_ns
                .iter()
                .skip(1)
                .take(4)
                .map(|&n| format!("{:.2}", ms(n)))
                .collect::<Vec<_>>()
                .join(" "),
            facts.prompt_tokens,
        );
    }
}

/// The prefill's and the first step's dispatch census, and the first steps'
/// phase split: populated only where the loop recorded them (a profile build,
/// `CANDLE_METAL_PROFILE=1`), so a plain run prints nothing here.
fn print_census_and_phases(stats: &DecodeStats) {
    for (name, census) in [
        ("prefill", &stats.prefill_census),
        ("step 0", &stats.step0_census),
    ] {
        if census.is_empty() {
            continue;
        }
        let total: u64 = census.iter().map(|(_, n)| n).sum();
        eprintln!(
            "[{name} census: {total} dispatches, {} kernels]",
            census.len()
        );
        for (label, n) in census {
            eprintln!("    {n:>6}  {label}");
        }
    }
    let ms = |ns: u64| ns as f64 / 1e6;
    for (i, p) in stats.first_steps.iter().enumerate() {
        eprintln!(
            "[step {i} phases: sample {:.2} ms, forward {:.2} ms, backpressure {:.2} ms, eos_wait {:.2} ms]",
            ms(p.sample_ns),
            ms(p.forward_ns),
            ms(p.backpressure_ns),
            ms(p.eos_ns)
        );
    }
}

/// The host step's shape (debug and profile builds, where `step_ns` is
/// populated): the phase totals, and the per-step time's percentiles.
fn print_step_shape(stats: &DecodeStats, verbose: bool) {
    let ms = |ns: u64| ns as f64 / 1e6;
    if stats.phase_steps > 0 {
        let n = stats.phase_steps as f64;
        let p: &StepPhases = &stats.phase_totals;
        let total = (p.sample_ns + p.forward_ns + p.backpressure_ns + p.eos_ns) as f64;
        eprintln!(
            "[hot thread per step, mean over {} steps: sample {:.3} ms, forward (encode) {:.3} ms, \
             backpressure {:.3} ms, eos_wait {:.3} ms; encoding {:.1} % of the bracketed step]",
            stats.phase_steps,
            ms(p.sample_ns) / n,
            ms(p.forward_ns) / n,
            ms(p.backpressure_ns) / n,
            ms(p.eos_ns) / n,
            100.0 * (p.sample_ns + p.forward_ns) as f64 / total.max(1.0),
        );
    }
    if verbose && stats.step_ns.len() > 1 {
        let mut v: Vec<f64> = stats.step_ns.iter().skip(1).map(|&ns| ms(ns)).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        let p50 = quantile(&v, 0.5);
        eprintln!(
            "[step time, host, steps 2..={} (the first step carries the prefill's wait): p50 {:.3} / mean {:.3} / p90 {:.3} / \
             p99 {:.3} / max {:.3} ms; mean/median {:.4}; steps over 1.5× the median: {}]",
            v.len() + 1,
            p50,
            mean,
            quantile(&v, 0.9),
            quantile(&v, 0.99),
            v.last().copied().unwrap_or(f64::NAN),
            mean / p50.max(1e-9),
            v.iter().filter(|&&x| x > 1.5 * p50).count(),
        );
    }
}

/// A speculative run: a "step" is a ROUND, which publishes 1..K+1 tokens, so
/// the token rate is rounds/s times tokens/round; both are reported rather
/// than one relabelled as the other. Then the per-round breakdown.
fn print_speculative(
    stats: &DecodeStats,
    tokens: usize,
    wall: std::time::Duration,
    k: usize,
    wanted: Wanted,
) {
    if wanted.verbose {
        let rounds = stats.steps.max(1);
        // The stream minus the prefill's anchor token, emitted before round 0.
        let round_tokens = tokens.saturating_sub(usize::from(tokens > 0));
        let tokens_per_round = round_tokens as f64 / rounds as f64;
        let decode_s = wall.as_secs_f64() - stats.prefill_ns as f64 / 1e9;
        let rounds_per_s = if decode_s > 0.0 {
            rounds as f64 / decode_s
        } else {
            f64::NAN
        };
        eprintln!(
            "[dspark: {tokens_per_round:.2} accepted-tokens/round (mean over {rounds} rounds), \
             {:.1} tok/s steady (rounds/s × tokens/round), K={k}]",
            rounds_per_s * tokens_per_round,
        );
        // The drafter's own context against the target's position: equal
        // when it followed the request's prefix reuse (issue #894).
        if let Some((ctx, pos)) = stats.drafter_ctx {
            eprintln!(
                "[dspark drafter context: {ctx} positions, target at {pos}{}]",
                if ctx == pos {
                    ""
                } else {
                    " — NOT FOLLOWING THE TARGET"
                }
            );
        }
    }
    print_rounds(stats, wanted);
}

/// The per-round breakdown of a speculative run: where a round's wall goes,
/// per round and per accepted token. The first rounds are dropped — they pay
/// the drafter's pipeline construction and a cold pool — and the count is
/// printed. Wall figures are medians per round and ratios of sums per token
/// (§10.2j); GPU-busy and dispatch figures appear only under `phase_sync`
/// with `CANDLE_METAL_PROFILE=1`, and the census is one round's.
pub fn print_rounds(stats: &DecodeStats, wanted: Wanted) {
    const WARMUP_ROUNDS: usize = 4;
    let phase_sync = wanted.phase_sync;
    let rounds = &stats.rounds;
    if rounds.is_empty() {
        return;
    }
    // Acceptance over every round (the warm-up's included: it is the
    // drafter's, not the timing's), `hist[a]` the rounds that kept `a`
    // tokens, the bonus counted. Reported however short the run: a short
    // answer's acceptance is a gate's reading too.
    let widest = rounds.iter().map(|r| r.accepted).max().unwrap_or(0);
    let hist: Vec<usize> = (0..=widest)
        .map(|a| rounds.iter().filter(|r| r.accepted == a).count())
        .collect();
    let all: usize = rounds.iter().map(|r| r.accepted).sum();
    eprintln!(
        "DSPARK_ACCEPT rounds={} tokens={all} tokens_per_round={:.4} accepted_hist={hist:?}",
        rounds.len(),
        all as f64 / rounds.len() as f64
    );
    if rounds.len() <= WARMUP_ROUNDS {
        if wanted.verbose {
            eprintln!(
                "[dspark rounds: only {} rounds, fewer than the {WARMUP_ROUNDS} dropped as warm-up; \
                 no breakdown]",
                rounds.len()
            );
        }
        return;
    }
    let timed = &rounds[WARMUP_ROUNDS..];
    let n = timed.len() as f64;
    let ms = |ns: u64| ns as f64 / 1e6;
    let median = |mut v: Vec<f64>| -> f64 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = v.len();
        if n % 2 == 1 {
            v[n / 2]
        } else {
            0.5 * (v[n / 2 - 1] + v[n / 2])
        }
    };
    let accepted: usize = timed.iter().map(|r| r.accepted).sum();
    let tokens_per_round = accepted as f64 / n;
    let sum_draft: f64 = timed.iter().map(|r| ms(r.timing.draft_ns)).sum();
    let sum_verify: f64 = timed.iter().map(|r| ms(r.timing.verify_ns)).sum();
    let sum_observe: f64 = timed.iter().map(|r| ms(r.timing.observe_ns)).sum();
    let sum_round = sum_draft + sum_verify + sum_observe;
    let med_round = median(timed.iter().map(|r| ms(r.timing.wall_ns())).collect());
    let med_draft = median(timed.iter().map(|r| ms(r.timing.draft_ns)).collect());
    let med_verify = median(timed.iter().map(|r| ms(r.timing.verify_ns)).collect());
    let med_observe = median(timed.iter().map(|r| ms(r.timing.observe_ns)).collect());
    let e = accepted as f64;
    let rb_draft: u64 = timed.iter().map(|r| r.timing.draft_readbacks).sum();
    let rb_verify: u64 = timed.iter().map(|r| r.timing.verify_readbacks).sum();
    let rb_observe: u64 = timed.iter().map(|r| r.timing.observe_readbacks).sum();
    let gpu_draft: f64 = timed.iter().map(|r| r.timing.draft_gpu_s * 1e3).sum();
    let gpu_verify: f64 = timed.iter().map(|r| r.timing.verify_gpu_s * 1e3).sum();
    let gpu_observe: f64 = timed.iter().map(|r| r.timing.observe_gpu_s * 1e3).sum();
    let disp_draft: u64 = timed.iter().map(|r| r.timing.draft_disp).sum();
    let disp_verify: u64 = timed.iter().map(|r| r.timing.verify_disp).sum();
    let disp_observe: u64 = timed.iter().map(|r| r.timing.observe_disp).sum();
    let profiled = disp_draft + disp_verify + disp_observe > 0;
    if wanted.verbose {
        eprintln!(
            "[dspark rounds: {} timed ({WARMUP_ROUNDS} warm-up dropped), {accepted} tokens accepted, \
             {tokens_per_round:.3} tokens/round; phase_sync={phase_sync}]",
            timed.len()
        );
        eprintln!("[dspark per round, ms (median | mean):  draft {med_draft:.3} | {:.3}   verify {med_verify:.3} | {:.3}   observe {med_observe:.3} | {:.3}   round {med_round:.3} | {:.3}]",
            sum_draft / n, sum_verify / n, sum_observe / n, sum_round / n);
        eprintln!("[dspark per accepted token, ms (ratio of sums):  draft {:.3}   verify {:.3}   observe {:.3}   total {:.3}]",
            sum_draft / e, sum_verify / e, sum_observe / e, sum_round / e);
        // The round's cost is paid whatever the acceptance, so the accepted
        // length is the other factor in the per-token figure.
        let mut hist = vec![0usize; timed.iter().map(|r| r.accepted).max().unwrap_or(0) + 1];
        for r in timed {
            hist[r.accepted] += 1;
        }
        eprintln!(
            "[dspark accepted-per-round histogram (1..): {}]",
            hist.iter()
                .enumerate()
                .skip(1)
                .map(|(k, c)| format!("{k}:{c}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
        // Each readback is a wait for the GPU, and a round owes exactly one
        // (the verify's argmax): counted, since a draft that drained its ids
        // to the host moves no digest.
        eprintln!(
            "[dspark host readbacks per round (mean):  draft {:.2}   verify {:.2}   observe {:.2}   round {:.2}]",
            rb_draft as f64 / n,
            rb_verify as f64 / n,
            rb_observe as f64 / n,
            (rb_draft + rb_verify + rb_observe) as f64 / n
        );
        // What the steady-state round allocated: the issue #894 gate reads
        // zero of both after warm-up (the verify is a planned pass, the
        // drafter's context is owned pages), and the round's spread beside
        // it, since the churn is the noise source the gate is for.
        let allocs: u64 = timed.iter().map(|r| r.timing.pool_allocations).sum();
        let adds: u64 = timed.iter().map(|r| r.timing.residency_adds).sum();
        let twins: u64 = timed
            .iter()
            .map(|r| r.timing.pool_allocations_pending_twin)
            .sum();
        let allocating = timed
            .iter()
            .filter(|r| r.timing.pool_allocations > 0 || r.timing.residency_adds > 0)
            .count();
        eprintln!(
            "[dspark allocations after warm-up: pool allocations={allocs} (of which pending-twin={twins}) \
             residency adds={adds} over {} rounds, {allocating} rounds allocated anything]",
            timed.len()
        );
        let mut walls: Vec<f64> = timed.iter().map(|r| ms(r.timing.wall_ns())).collect();
        walls.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        eprintln!(
            "[dspark round ms after warm-up: p50 {:.3} p90 {:.3} p99 {:.3} max {:.3}]",
            quantile(&walls, 0.5),
            quantile(&walls, 0.9),
            quantile(&walls, 0.99),
            quantile(&walls, 1.0)
        );
        if profiled {
            eprintln!("[dspark GPU-busy per round, ms (mean of union):  draft {:.3}   verify {:.3}   observe {:.3}   round {:.3}  — idle (wall−busy) {:.3}]",
                gpu_draft / n, gpu_verify / n, gpu_observe / n, (gpu_draft + gpu_verify + gpu_observe) / n,
                (sum_round - gpu_draft - gpu_verify - gpu_observe) / n);
            eprintln!(
                "[dspark dispatches per round (mean):  draft {:.1}   verify {:.1}   observe {:.1}]",
                disp_draft as f64 / n,
                disp_verify as f64 / n,
                disp_observe as f64 / n
            );
        }
    }
    if profiled {
        for (name, census) in [
            (
                "draft",
                timed.iter().find_map(|r| r.timing.draft_census.as_ref()),
            ),
            (
                "verify",
                timed.iter().find_map(|r| r.timing.verify_census.as_ref()),
            ),
            (
                "observe",
                timed.iter().find_map(|r| r.timing.observe_census.as_ref()),
            ),
        ] {
            if let Some(c) = census {
                let total: u64 = c.iter().map(|(_, n)| n).sum();
                eprintln!(
                    "[dspark census {name} (one round, {total} dispatches, {} labels):",
                    c.len()
                );
                for (label, count) in c {
                    eprintln!("    {count:>6}  {label}");
                }
                eprintln!("]");
            }
        }
    }
    eprintln!(
        "DSPARK_ROUNDS rounds={} tokens={accepted} tokens_per_round={tokens_per_round:.4} \
         round_ms_median={med_round:.4} round_ms_mean={:.4} \
         draft_ms_median={med_draft:.4} verify_ms_median={med_verify:.4} observe_ms_median={med_observe:.4} \
         draft_ms_per_token={:.4} verify_ms_per_token={:.4} observe_ms_per_token={:.4} total_ms_per_token={:.4} \
         draft_gpu_ms={:.4} verify_gpu_ms={:.4} observe_gpu_ms={:.4} \
         draft_disp={:.1} verify_disp={:.1} observe_disp={:.1} \
         draft_readbacks={:.3} verify_readbacks={:.3} observe_readbacks={:.3} \
         phase_sync={phase_sync} profiled={profiled}",
        timed.len(),
        sum_round / n,
        sum_draft / e,
        sum_verify / e,
        sum_observe / e,
        sum_round / e,
        gpu_draft / n,
        gpu_verify / n,
        gpu_observe / n,
        disp_draft as f64 / n,
        disp_verify as f64 / n,
        disp_observe as f64 / n,
        rb_draft as f64 / n,
        rb_verify as f64 / n,
        rb_observe as f64 / n,
    );
}

/// The memory gauge's peaks over a run, in the `MEMPEAK` spelling: exact,
/// load-independent bytes; the sample count says the sampler ran. `first` is
/// the reading before the run (the floor: the weights and the pool's retained
/// buffers), so the excursion is `peak − first`.
pub fn print_mempeak(m: &MemReport) {
    let Some(p) = &m.peak else { return };
    let gb = |b: u64| b as f64 / 1e9;
    let first = m.before.device_allocated;
    eprintln!(
        "[mem: {} samples; device_allocated first {:.3} GB peak {:.3} GB (+{:.3}); pool peaks \
         live {:.3} free {:.3} pending {:.3} GB; phys_footprint peak {:.3} GB]",
        m.samples,
        gb(first),
        gb(p.device_allocated),
        gb(p.device_allocated.saturating_sub(first)),
        gb(p.pool_live),
        gb(p.pool_free),
        gb(p.pool_pending),
        gb(p.phys_footprint),
    );
    eprintln!(
        "MEMPEAK samples={} first_device_gb={:.4} peak_device_gb={:.4} device_excursion_gb={:.4} \
         peak_pool_live_gb={:.4} peak_pool_free_gb={:.4} peak_pool_pending_gb={:.4} \
         peak_phys_gb={:.4}",
        m.samples,
        gb(first),
        gb(p.device_allocated),
        gb(p.device_allocated.saturating_sub(first)),
        gb(p.pool_live),
        gb(p.pool_free),
        gb(p.pool_pending),
        gb(p.phys_footprint),
    );
}

/// The engagement of every installed executor arm: whether the ICB executor
/// replayed and what it covered, whether the arena installed, whether the
/// packed parameters stayed pooled, what the pass-plan cache served. An arm
/// selected is not an arm engaged; `replaying=false` or `covered=0` means an
/// A/B compared the classical path against itself wearing the arm's name.
pub fn print_executor(device: &Device, installed: &Installed, steps: u64) {
    if !installed.any() {
        return;
    }
    #[cfg(feature = "metal")]
    {
        if let Some(exec) = &installed.replay {
            print_replay(exec, steps);
        }
        if installed.packed && !installed.replay_installed() {
            let (slots, c) = lloom_tensor::kernels::constants_pool()
                .map(|p| (p.len(), p.counters()))
                .unwrap_or_default();
            eprintln!(
                "[packed: constants_pool slots={slots} allocated={} unchanged={} rewritten={} \
                 detached={}]",
                c.allocated, c.unchanged, c.rewritten, c.detached,
            );
        }
        if let Device::Metal(dev) = device {
            if installed.pass_plans {
                print_pass_plans(dev, installed.pass_widths);
            }
            if installed.arena {
                match dev.arena_auto_report() {
                    Some(r) => eprintln!(
                        "[arena: {:?} installed after decode step {}: {} of {} ordinals served; \
                         hazard_key={:?} rerecordings={} counters={:?}]",
                        r.layout,
                        r.installed_at_step,
                        r.covered,
                        r.total,
                        lloom_tensor::metal_backend::hazard_key(),
                        dev.arena_rerecordings(),
                        dev.arena_counters(),
                    ),
                    None => eprintln!("[arena: not installed]"),
                }
            }
        }
    }
    #[cfg(not(feature = "metal"))]
    let _ = (device, steps);
}

/// The pass-plan cache's counters, and each key's coverage — with the
/// escapes' identities (ordinal, bytes, allocated-at, first-read-at), since a
/// count says a hole exists and the sizes say what leaked. Beside them, what
/// the served passes still asked the pools for: a key's coverage cannot show
/// that, because the recorder sees only the private pool (`DESIGN.md`
/// §9.2aa, I1).
#[cfg(feature = "metal")]
pub fn print_pass_plans(dev: &lloom_tensor::MetalDevice, widths: &str) {
    let (Some(c), Some((keys, base))) = (dev.pass_plan_counters(), dev.pass_plan_report()) else {
        return;
    };
    let keys: Vec<String> = keys
        .iter()
        .map(|(k, covered, total, escapes)| {
            // The kind, then the model's sequence variant and a model-opened
            // pass's mark when they are not the default (`PassKey`).
            let mut name = format!("{:?}", k.kind);
            if k.variant != 0 {
                name.push_str(&format!("#v{}", k.variant));
            }
            if k.opener == lloom_tensor::metal_backend::PassOpener::Model {
                name.push_str("@model");
            }
            if *escapes > 0 {
                format!(
                    "{name}={covered}/{total}(+{escapes} escaped {:?})",
                    dev.pass_plan_escapes(k)
                )
            } else {
                format!("{name}={covered}/{total}")
            }
        })
        .collect();
    let (shared, private) = dev.served_pass_pool_requests();
    eprintln!(
        "[pass plans: served={} recorded={} observed={} plans={} evicted={} \
         escapes={} capped={} base_grows={} served_pass_pool_requests=shared:{shared},private:{private} \
         base={:.1} MiB widths={} keys=[{}]]",
        c.served,
        c.recorded,
        c.observed,
        c.plans,
        c.evicted,
        c.escapes,
        c.capped,
        c.base_grows,
        base as f64 / (1024.0 * 1024.0),
        widths,
        keys.join(" ")
    );
    // `LLOOM_SERVED_REQUEST_TRACE`: the served passes' requests by site.
    for (pool, site, n) in lloom_tensor::metal_backend::served_pass_request_sites() {
        eprintln!(
            "[served pass requests: {n} {} {site}]",
            if pool == 0 { "shared" } else { "private" }
        );
    }
}

/// What the ICB executor did: engagement first, then the residency term
/// (`DESIGN.md` §11.3w), the constants pool's arms (`detached > 0` is the
/// check that a lookahead loop never rewrote a params block under an
/// in-flight reader; `rewritten` past the first steps is a position whose
/// block changes between steps), and what went stale, attributed.
#[cfg(feature = "metal")]
fn print_replay(exec: &lloom_tensor::kernels::IcbExecutor, steps: u64) {
    use lloom_tensor::kernels::metal::trace;
    let cov = exec.coverage();
    eprintln!(
        "[replay: replaying={} covered={}/{} runs={} barriers_encoded={} stale={} \
         stale_runs={} prefix_flushes={} rerecordings={} discarded_windows={} \
         excluded_varies={} excluded_inline_consts={} excluded_no_pipeline={} excluded_quarantined={}]",
        exec.is_replaying(),
        cov.covered,
        cov.positions,
        exec.runs(),
        exec.encoded_barriers(),
        exec.stale_positions(),
        exec.stale_runs(),
        exec.prefix_flushes(),
        exec.rerecordings(),
        exec.discarded_windows(),
        cov.varies,
        cov.inline_constants,
        cov.no_pipeline,
        cov.quarantined,
    );
    eprintln!(
        "[replay: warm_in_skipped={} generations_built={} pipeline_refreshes={} skipped_steps={}]",
        exec.warm_in_skipped(),
        exec.generations_built(),
        exec.pipeline_refreshes(),
        exec.skipped_steps()
    );
    let ranges = trace::execute_range_count();
    let use_resource = trace::use_resource_count();
    let per = steps.max(1) as f64;
    // Named from the plan, not from the variable that selected it (§7.1a-i):
    // per-run declares `residency_per_step`, plan-wide `runs × plan_wide`.
    let arm = if exec.runs() > 0 && exec.residency_per_step() == exec.runs() * exec.plan_resident()
    {
        "plan-wide"
    } else {
        "per-run"
    };
    eprintln!(
        "[replay: execute_ranges={ranges} ({:.1}/step) useResource={use_resource} ({:.1}/step) \
         residency={arm} per_step={} plan_wide={} barriers_suppressed={}]",
        ranges as f64 / per,
        use_resource as f64 / per,
        exec.residency_per_step(),
        exec.plan_resident(),
        trace::barriers_suppressed(),
    );
    let pool = lloom_tensor::kernels::constants_pool()
        .map(|p| (p.len(), p.counters()))
        .unwrap_or_default();
    eprintln!(
        "[replay: constants_pool slots={} allocated={} unchanged={} rewritten={} detached={}]",
        pool.0, pool.1.allocated, pool.1.unchanged, pool.1.rewritten, pool.1.detached,
    );
    for ((kernel, reason), n) in &cov.excluded_by_kernel {
        eprintln!("[replay excluded {n:4} {kernel:<44} {reason}]");
    }
    let stale = exec.stale_report();
    if !stale.first.is_empty() {
        let by_axis: Vec<String> = stale
            .by_axis
            .iter()
            .map(|(axis, n)| format!("{axis}={n}"))
            .collect();
        eprintln!(
            "[replay stale: steps_with_stale={} by_axis: {}]",
            stale.steps,
            by_axis.join(" ")
        );
        for (kernel, n) in &stale.by_kernel {
            eprintln!("[replay stale {n:4} {kernel}]");
        }
        for (step, pos, kernel, axis, at_head) in &stale.first {
            eprintln!(
                "[replay stale first: step={step} pos={pos} {kernel} {axis} {}]",
                if *at_head { "head" } else { "member" }
            );
        }
    }
}

/// The allocator-side counters a run moves, summed over the shared and the
/// private pool.
#[cfg(feature = "metal")]
#[derive(Clone, Copy, Default)]
struct ChurnSnapshot {
    lookups: u64,
    hits: u64,
    allocations: u64,
    evicted: u64,
    trims: u64,
    res_added: u64,
    res_removed: u64,
    res_commits: u64,
    to_cpu_fast: u64,
    to_cpu_blit: u64,
    to_cpu_null: u64,
    to_cpu_staged: u64,
    readbacks: lloom_tensor::metal_backend::ReadbackCounts,
    /// `(barriers, encoder sessions, command buffers)` (#814 item B).
    shape: (u64, u64, u64),
}

#[cfg(feature = "metal")]
fn churn_snapshot(m: &lloom_tensor::MetalDevice) -> ChurnSnapshot {
    let (shared, private) = m.pool_counters();
    let res = m.residency_counters();
    let (fast, blit) = lloom_tensor::metal_backend::to_cpu_counts();
    ChurnSnapshot {
        lookups: shared.lookups + private.lookups,
        hits: shared.hits + private.hits,
        allocations: shared.allocations + private.allocations,
        evicted: shared.evicted + private.evicted,
        trims: shared.trims + private.trims,
        res_added: res.added,
        res_removed: res.removed,
        res_commits: res.commits,
        to_cpu_fast: fast,
        to_cpu_blit: blit,
        to_cpu_null: lloom_tensor::metal_backend::to_cpu_null_after_wait(),
        to_cpu_staged: lloom_tensor::metal_backend::to_cpu_staged_count(),
        readbacks: lloom_tensor::metal_backend::readback_counts(),
        shape: lloom_tensor::kernels::metal::trace::shape_counts(),
    }
}

/// The run's churn as deltas — pool misses, evictions, residency adds and
/// commits, the readbacks split fast / blit / null-after-wait — then what
/// sits in the free budget at the end, which names the sizes an allocation
/// count growing with `n` is evicted to make room for.
#[cfg(feature = "metal")]
fn print_churn(m: &lloom_tensor::MetalDevice, before: &ChurnSnapshot) {
    let after = churn_snapshot(m);
    eprintln!(
        "[churn over the run: pool lookups={} hits={} allocations={} evicted={} \
         trims={} | residency adds={} removes={} commits={} | to_cpu fast={} blit={} \
         null_after_wait={} staged={} staged_readback={} slot={} staging_allocs={} | \
         barriers={} encoder_sessions={} command_buffers={}]",
        after.lookups - before.lookups,
        after.hits - before.hits,
        after.allocations - before.allocations,
        after.evicted - before.evicted,
        after.trims - before.trims,
        after.res_added - before.res_added,
        after.res_removed - before.res_removed,
        after.res_commits - before.res_commits,
        after.to_cpu_fast - before.to_cpu_fast,
        after.to_cpu_blit - before.to_cpu_blit,
        after.to_cpu_null - before.to_cpu_null,
        after.to_cpu_staged - before.to_cpu_staged,
        after.readbacks.staged_readback - before.readbacks.staged_readback,
        after.readbacks.slot - before.readbacks.slot,
        after.readbacks.staging_allocs - before.readbacks.staging_allocs,
        after.shape.0 - before.shape.0,
        after.shape.1 - before.shape.1,
        after.shape.2 - before.shape.2,
    );
    // `LLOOM_READBACK_TRACE`: every readback of the run by kind and site.
    for (kind, site, n) in lloom_tensor::metal_backend::readback_sites() {
        eprintln!("[readback sites: {n} {kind} {site}]");
    }
    let (shared, private) = m.pool_free_summary(6);
    let fmt = |s: &lloom_tensor::metal_backend::FreeSummary| {
        let top: Vec<String> = s
            .top
            .iter()
            .map(|(size, n)| format!("{:.1}MB×{n}", *size as f64 / 1048576.0))
            .collect();
        format!(
            "{:.0}/{:.0} MB in {} buffers, top {}",
            s.bytes as f64 / 1048576.0,
            s.budget as f64 / 1048576.0,
            s.buffers,
            top.join(" ")
        )
    };
    let clock = m.gpu_clock();
    let (c_hits, c_misses) = m.constants_counters();
    eprintln!(
        "[pool free at end: shared {} | private {}; clock epochs submitted={} \
         completed={}; constants cache hits={c_hits} misses={c_misses}]",
        fmt(&shared),
        fmt(&private),
        clock.current_epoch(),
        clock.completed_count()
    );
    eprintln!(
        "[ledger: {} device_allocated={:.2} GB]",
        m.ledger().snapshot().line(),
        m.device().current_allocated_size() as f64 / 1e9
    );
    // The unretained arm's engagement: what the host held for command buffers
    // that hold nothing themselves (`in_flight.rs`).
    match m.in_flight_hold_counts() {
        Some(h) => eprintln!(
            "[command buffers: unretained held fences={} buffers={} pending={}]",
            h.fences, h.buffers, h.pending
        ),
        None => eprintln!("[command buffers: retaining]"),
    }
}

/// The dispatch recorder's view of the run (`CANDLE_METAL_PROFILE=1`): GPU
/// time by kernel from each command buffer's last label, the busy union
/// against the run's wall, the command-buffer lifecycle priced on the hot
/// thread, and the idle between buffers. The caller has synchronised, so
/// every completion handler has run.
#[cfg(feature = "metal")]
fn print_profile(stats: &DecodeStats, wall: std::time::Duration, call: std::time::Duration) {
    let s = lloom_tensor::kernels::metal::profile::snapshot();
    let steps = stats.steps.max(1) as f64;
    // Exact only at `CANDLE_METAL_COMPUTE_PER_BUFFER=1` (one dispatch per
    // buffer); at the default period a buffer is charged to its last kernel,
    // and the `buffers` column says so.
    {
        let by = lloom_tensor::kernels::metal::profile::gpu_time_by_label_detailed();
        let total: f64 = by.iter().map(|k| k.total_s).sum();
        eprintln!(
            "[profile: gpu time by kernel ({} labels, {:.3} ms/step summed over {:.0} buffers/step; \
             exact iff one dispatch per buffer. **p50 is the honest per-buffer figure**: a buffer's \
             interval bounds its RESIDENCY, so max >> p50 means that buffer absorbed a host wait and \
             this label's total is the wait, not the kernel — lloom #727):",
            by.len(),
            1e3 * total / steps,
            by.iter().map(|k| k.buffers as f64).sum::<f64>() / steps
        );
        for k in by.iter().take(24) {
            // Two shapes: a buffer resident for milliseconds on EVERY step
            // (tight distribution, high p50) holds something open across the
            // step — a host round-trip, a sync — and its total is that hold;
            // a single buffer absorbing one wait shows as max >> p50.
            let held = k.p50_s > 1e-3;
            let spiky = k.max_s > 8.0 * k.p50_s.max(1e-9) && k.max_s > 1e-3;
            let flag = if held {
                "  <-- HELD OPEN every step: residency, not compute"
            } else if spiky {
                "  <-- one buffer absorbed a wait"
            } else {
                ""
            };
            eprintln!(
                "    {:>8.3} ms/step  {:>6.1} buf/step  p50 {:>7.1} / p99 {:>8.1} / max {:>8.1} µs/buf  {}{}",
                1e3 * k.total_s / steps,
                k.buffers as f64 / steps,
                1e6 * k.p50_s,
                1e6 * k.p99_s,
                1e6 * k.max_s,
                k.label,
                flag
            );
        }
        eprintln!("]");
    }
    eprintln!(
        "[profile: gpu_busy {:.3} ms/step (union over {} command buffers, {} timed), \
         {:.1} dispatches/step, {} steps]",
        1e3 * s.gpu_busy_union_s / steps,
        s.command_buffers,
        s.timed_command_buffers,
        s.dispatches as f64 / steps,
        stats.steps,
    );
    // `1 − busy/wall` bounds the time no command buffer was executing from
    // above: a buffer stalled on a fence inside its interval counts as busy.
    let wall = wall.as_secs_f64();
    eprintln!(
        "[profile: gpu_busy_union {:.4} s / decode_wall {:.4} s = {:.4} busy/wall; \
         gpu_span {:.4} s; gpu_busy_sum {:.4} s; decode_wall runs to the device's drain, the \
         call returned at {:.4} s]",
        s.gpu_busy_union_s,
        wall,
        s.gpu_busy_union_s / wall.max(1e-9),
        s.gpu_span_s,
        s.gpu_busy_sum_s,
        call.as_secs_f64(),
    );
    if lloom_tensor::kernels::metal::profile::counters_enabled() {
        print_encoder_counters(&s);
    }
    eprintln!(
        "[profile: cb lifecycle on the hot thread: {:.1} commits/step at {:.1} µs each = \
         {:.3} ms/step (of which commandBuffer() {:.3} ms/step); {:.2} prunes/step at \
         {:.1} µs each = {:.3} ms/step]",
        s.commits as f64 / steps,
        s.commit_ns as f64 / 1e3 / s.commits.max(1) as f64,
        s.commit_ns as f64 / 1e6 / steps,
        s.create_cb_ns as f64 / 1e6 / steps,
        s.prunes as f64 / steps,
        s.prune_ns as f64 / 1e3 / s.prunes.max(1) as f64,
        s.prune_ns as f64 / 1e6 / steps,
    );
    print_idle_gaps(&s, stats, wall);
}

/// Per-kernel GPU time from the counter-sampled encoders
/// (`CANDLE_METAL_PROFILE_COUNTERS=1`), decode steps only.
///
/// Printed with the checks that make it readable: how many encoders were
/// sampled, dropped (with any sample buffer the device refused), resolved to
/// the error marker, empty, or never resolved; how many fall outside their
/// command buffer (the clock and unit check — zero is the only readable
/// answer); and what the encoders cover against the command buffers' union. A
/// kernel's row is exact only when its encoders hold one dispatch each
/// (`CANDLE_METAL_PROFILE_ENCODER_PER_DISPATCH=1`); otherwise the time sits in
/// the `<encoders of several dispatches>` row.
///
/// A GEMV's row carries its weight MB and GB/s — the largest input it bound
/// over its time; no other kernel's largest input is its traffic, so none
/// prints one.
///
/// `CANDLE_METAL_PROFILE_DUMP=<path>` adds `<path>.encoders.csv`, one row per
/// encoder.
#[cfg(feature = "metal")]
fn print_encoder_counters(s: &lloom_tensor::kernels::metal::profile::Snapshot) {
    use lloom_tensor::kernels::metal::profile as p;
    if let Some(why) = &s.counters_unavailable {
        eprintln!("[profile: encoder counters OFF: {why}]");
        return;
    }
    let decode: Vec<p::EncoderRecord> = s
        .encoder_records
        .iter()
        .filter(|r| r.step >= 1)
        .cloned()
        .collect();
    let steps = {
        let mut v: Vec<u64> = decode.iter().map(|r| r.step).collect();
        v.sort_unstable();
        v.dedup();
        v.len().max(1) as f64
    };
    // The command buffers committed in the same decode steps: the
    // denominator the encoders are reconciled against.
    let cb_union_decode = {
        let mut iv: Vec<(f64, f64)> = s
            .command_buffer_records
            .iter()
            .filter(|r| r.step >= 1)
            .map(|r| (r.interval.start, r.interval.end))
            .collect();
        iv.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let (mut u, mut cur) = (0.0, None::<(f64, f64)>);
        for (a, b) in iv {
            cur = match cur {
                Some((x, y)) if a <= y => Some((x, y.max(b))),
                Some((x, y)) => {
                    u += y - x;
                    Some((a, b))
                }
                None => Some((a, b)),
            };
        }
        u + cur.map_or(0.0, |(x, y)| y - x)
    };
    let enc_sum: f64 = decode.iter().map(|r| r.end - r.start).sum();
    let enc_union = p::encoder_union_s(&decode);
    let outside = p::encoder_containment(&s.encoder_records, 1e-6);
    let single = decode.iter().filter(|r| r.dispatches == 1).count();
    let refused = match &s.counter_refusal {
        Some(why) => format!(
            " ({} sample buffers refused, last with {why})",
            s.counter_buffers_refused
        ),
        None => String::new(),
    };
    eprintln!(
        "[profile: encoder counters: {} sampled ({} in decode steps, {} holding one dispatch), \
         dropped {}{refused}, error-marker {}, empty {}, unresolved buffers {}, sample unit {}, \
         OUTSIDE their command buffer {}; sample buffers {} (peak {} command buffers in flight, \
         slowest allocation {:.1} µs); decode: encoders sum {:.3} ms/step, union {:.3} ms/step, \
         against the command buffers' union {:.3} ms/step ({:.2} %), {:.0} steps]",
        s.encoder_records.len(),
        decode.len(),
        single,
        s.encoder_samples_dropped,
        s.encoder_samples_bad,
        s.encoder_samples_empty,
        s.encoder_buffers_unresolved,
        match s.counter_unit_s {
            Some(u) if (u - 1e-9).abs() < 1e-15 => "1 ns".to_string(),
            Some(u) => format!("{:.3} ns", u * 1e9),
            None => "unresolved".to_string(),
        },
        outside,
        s.counter_buffers_live,
        s.encoder_buffers_in_flight_peak,
        1e6 * s.counter_alloc_max_s,
        1e3 * enc_sum / steps,
        1e3 * enc_union / steps,
        1e3 * cb_union_decode / steps,
        100.0 * enc_union / cb_union_decode.max(1e-12),
        steps,
    );
    eprintln!(
        "[profile: gpu time by kernel from encoder counters (decode steps; share of the \
         encoders' sum; weight MB and GB/s for a GEMV only, from the largest input it bound):"
    );
    for k in p::encoder_time_by_label(&decode) {
        let weights = if k.label.starts_with("gemv") {
            format!(
                "{:>9.3} MB/step {:>7.1} GB/s",
                k.input_bytes as f64 / 1e6 / steps,
                k.input_bytes as f64 / k.total_s.max(1e-12) / 1e9
            )
        } else {
            format!("{:>9} MB/step {:>7} GB/s", "-", "-")
        };
        eprintln!(
            "    {:>8.3} ms/step {:>6.1} %  {:>6.1} enc/step  p50 {:>8.2} / p90 {:>8.2} / max {:>9.2} µs  \
             {weights}  {}",
            1e3 * k.total_s / steps,
            100.0 * k.total_s / enc_sum.max(1e-12),
            k.encoders as f64 / steps,
            1e6 * k.p50_s,
            1e6 * k.p90_s,
            1e6 * k.max_s,
            k.label,
        );
    }
    eprintln!("]");
    if let Some(path) = lloom_tensor::tuning::tuning()
        .instrument
        .profile_dump
        .clone()
    {
        use std::io::Write;
        let path = format!("{path}.encoders.csv");
        let write = || -> std::io::Result<()> {
            let mut f = std::io::BufWriter::new(std::fs::File::create(&path)?);
            writeln!(
                f,
                "step,start_s,end_s,label,dispatches,input_bytes,cb_start_s,cb_end_s"
            )?;
            for r in &s.encoder_records {
                writeln!(
                    f,
                    "{},{:.9},{:.9},{},{},{},{:.9},{:.9}",
                    r.step,
                    r.start,
                    r.end,
                    r.label.as_deref().unwrap_or(""),
                    r.dispatches,
                    r.input_bytes,
                    r.cb_start,
                    r.cb_end
                )?;
            }
            Ok(())
        };
        match write() {
            Ok(()) => eprintln!(
                "[profile: dumped {} encoders to {path}]",
                s.encoder_records.len()
            ),
            Err(e) => eprintln!("[profile: encoder dump to {path} failed: {e}]"),
        }
    }
}

/// The GPU's idle between consecutive command buffers, as a distribution and
/// attributed: whether it is spread evenly (the seam between every pair of
/// buffers, GPU-side) or concentrated in a few steps (a transient stall), and
/// for each gap whether the next buffer was committed after the previous one
/// ended (the host starved the GPU) or was already queued (the driver's seam,
/// which no host thread can shorten).
///
/// `CANDLE_METAL_PROFILE_DUMP=<path>` writes the raw series: every command
/// buffer (`<path>`: step, commit, start, end) and every step's idle beside
/// its host step time (`<path>.steps.csv`).
#[cfg(feature = "metal")]
fn print_idle_gaps(
    s: &lloom_tensor::kernels::metal::profile::Snapshot,
    st: &DecodeStats,
    wall: f64,
) {
    let gaps = &s.idle_gaps;
    let steps = st.steps.max(1) as f64;
    let idle_total: f64 = gaps.iter().map(|g| g.idle).sum();
    let starved_total: f64 = gaps.iter().map(|g| g.starved).sum();
    let mut per_gap: Vec<f64> = gaps.iter().map(|g| g.idle * 1e6).collect();
    per_gap.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    eprintln!(
        "[profile: gpu idle between command buffers: {} gaps over {} timed buffers, {:.3} ms = \
         {:.2} % of decode_wall ({:.1} µs/step); starved (next buffer committed after the previous \
         ended) {:.3} ms = {:.1} % of the idle, seam (already queued) {:.3} ms; per gap p50 {:.1} / \
         p90 {:.1} / p99 {:.1} / max {:.1} µs; clock check min(GPUStartTime − committed) {:+.1} µs]",
        gaps.len(),
        s.timed_command_buffers,
        1e3 * idle_total,
        100.0 * idle_total / wall.max(1e-9),
        1e6 * idle_total / steps,
        1e3 * starved_total,
        100.0 * starved_total / idle_total.max(1e-12),
        1e3 * (idle_total - starved_total),
        quantile(&per_gap, 0.5),
        quantile(&per_gap, 0.9),
        quantile(&per_gap, 0.99),
        per_gap.last().copied().unwrap_or(f64::NAN),
        1e6 * s.min_start_after_commit_s,
    );
    let edges_us = [
        20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0,
    ];
    let mut count = vec![0u64; edges_us.len() + 1];
    let mut sum_ms = vec![0f64; edges_us.len() + 1];
    for g in gaps {
        let us = g.idle * 1e6;
        let b = edges_us
            .iter()
            .position(|&e| us < e)
            .unwrap_or(edges_us.len());
        count[b] += 1;
        sum_ms[b] += g.idle * 1e3;
    }
    let mut hist = String::new();
    for b in 0..count.len() {
        let label = if b == 0 {
            format!("<{}", edges_us[0])
        } else if b == edges_us.len() {
            format!("≥{}", edges_us[b - 1])
        } else {
            format!("{}–{}", edges_us[b - 1], edges_us[b])
        };
        hist.push_str(&format!("{label}: {} ({:.3} ms) | ", count[b], sum_ms[b]));
    }
    eprintln!(
        "[profile: idle gap histogram, µs (count, ms): {}]",
        hist.trim_end_matches(" | ")
    );
    // The idle laid at the step whose buffer ended it; step 0 is everything
    // committed before the loop's first step (the prefill).
    let n = st.steps as usize + 1;
    let mut step_idle = vec![0f64; n];
    let mut step_starved = vec![0f64; n];
    for g in gaps {
        let i = (g.step as usize).min(n - 1);
        step_idle[i] += g.idle;
        step_starved[i] += g.starved;
    }
    if n > 1 {
        let mut idx: Vec<usize> = (1..n).collect();
        idx.sort_by(|&a, &b| {
            step_idle[b]
                .partial_cmp(&step_idle[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut sorted: Vec<f64> = idx.iter().map(|&i| step_idle[i] * 1e6).collect();
        sorted.reverse();
        let decode_idle: f64 = step_idle[1..].iter().sum();
        let top = ((n - 1) as f64 * 0.05).ceil().max(1.0) as usize;
        let top_idle: f64 = idx.iter().take(top).map(|&i| step_idle[i]).sum();
        let mean_us = 1e6 * decode_idle / (n - 1) as f64;
        let top5: Vec<String> = idx
            .iter()
            .take(5)
            .map(|&i| {
                format!(
                    "step {} {:.0} µs (starved {:.0})",
                    i,
                    1e6 * step_idle[i],
                    1e6 * step_starved[i]
                )
            })
            .collect();
        eprintln!(
            "[profile: idle per step, steps 1..={}: p50 {:.1} / p90 {:.1} / p99 {:.1} / max {:.1} µs, \
             mean {:.1}, mean/median {:.3}; the top {} steps (5 %) hold {:.1} % of the decode's idle; \
             steps with zero idle: {}; top 5: {}; step 0 (before the loop) idle {:.3} ms, starved {:.3} ms]",
            n - 1,
            quantile(&sorted, 0.5),
            quantile(&sorted, 0.9),
            quantile(&sorted, 0.99),
            sorted.last().copied().unwrap_or(f64::NAN),
            mean_us,
            mean_us / quantile(&sorted, 0.5).max(1e-9),
            top,
            100.0 * top_idle / decode_idle.max(1e-12),
            step_idle[1..].iter().filter(|&&x| x == 0.0).count(),
            top5.join(", "),
            1e3 * step_idle[0],
            1e3 * step_starved[0],
        );
    }
    if let Some(path) = lloom_tensor::tuning::tuning()
        .instrument
        .profile_dump
        .clone()
    {
        use std::io::Write;
        let write = || -> std::io::Result<()> {
            let mut f = std::io::BufWriter::new(std::fs::File::create(&path)?);
            writeln!(f, "step,committed_s,start_s,end_s,label")?;
            for r in &s.command_buffer_records {
                writeln!(
                    f,
                    "{},{:.9},{:.9},{:.9},{}",
                    r.step,
                    r.committed,
                    r.interval.start,
                    r.interval.end,
                    r.label.as_deref().unwrap_or("")
                )?;
            }
            let mut f =
                std::io::BufWriter::new(std::fs::File::create(format!("{path}.steps.csv"))?);
            writeln!(f, "step,idle_us,starved_us,step_ns")?;
            for i in 0..n {
                // `step_ns[k]` is step `k + 1`; step 0 has no host step time.
                let ns = if i >= 1 { st.step_ns.get(i - 1) } else { None };
                writeln!(
                    f,
                    "{},{:.3},{:.3},{}",
                    i,
                    1e6 * step_idle[i],
                    1e6 * step_starved[i],
                    ns.map(|n| n.to_string()).unwrap_or_default()
                )?;
            }
            Ok(())
        };
        match write() {
            Ok(()) => eprintln!(
                "[profile: dumped {} command buffers to {path}, {} steps to {path}.steps.csv]",
                s.command_buffer_records.len(),
                n
            ),
            Err(e) => eprintln!("[profile: dump to {path} failed: {e}]"),
        }
    }
}

/// Under `CANDLE_METAL_TRACE=1`: every kernel the run dispatched, how many
/// times and in which unit (threadgroups or threads), so an arm that needs
/// every dispatch to go one way (the EOS gate: indirect threadgroups) shows
/// its coverage. Whole-run counts; two runs a step apart give the per-step
/// census by difference.
#[cfg(feature = "metal")]
pub fn print_trace_census(steps: u64) {
    use lloom_tensor::kernels::metal::trace;
    if !trace::trace_requested() {
        return;
    }
    // Per kernel: dispatches, and how many of them were ordered after their
    // predecessor (the barrier count that names a launch on the serial chain).
    let mut census: std::collections::BTreeMap<(String, bool), (u64, u64)> = Default::default();
    // Which GEMVs are ORDERED after their predecessor — a barrier before
    // them, or the first dispatch of an encoder session — and so pay a drain
    // and a fill, against those free to overlap it; keyed by weight bytes
    // (the largest input bound). Observed from the trace, not modelled.
    let mut ordering: std::collections::BTreeMap<usize, (u64, u64)> = Default::default();
    let (mut prev_encoder, mut ordered_all, mut n_all) = (u64::MAX, 0u64, 0u64);
    let mut flow = OutputFlow::default();
    for d in trace::take_dispatches() {
        flow.dispatch(&d);
        let ordered = d.barrier || d.encoder != prev_encoder;
        prev_encoder = d.encoder;
        n_all += 1;
        ordered_all += u64::from(ordered);
        if d.pipeline.starts_with("gemv") {
            let w = d
                .bindings
                .iter()
                .filter(|b| !b.is_output)
                .map(|b| b.len)
                .max()
                .unwrap_or(0);
            let e = ordering.entry(w).or_default();
            if ordered {
                e.0 += 1;
            } else {
                e.1 += 1;
            }
        }
        let c = census.entry((d.pipeline, d.by_threadgroups)).or_default();
        c.0 += 1;
        c.1 += u64::from(ordered);
    }
    let per = steps.max(1) as f64;
    eprintln!(
        "[trace ordering: {:.1} of {:.1} dispatches/step ordered after their predecessor (a \
         barrier or a new encoder); GEMVs by weight bytes, ordered / free to overlap, per step:",
        ordered_all as f64 / per,
        n_all as f64 / per
    );
    for (w, (o, f)) in &ordering {
        eprintln!(
            "  {:>10.3} MB: {:>6.1} ordered {:>6.1} free",
            *w as f64 / 1e6,
            *o as f64 / per,
            *f as f64 / per
        );
    }
    eprintln!("]");
    let total: u64 = census.values().map(|c| c.0).sum();
    let threads: u64 = census
        .iter()
        .filter(|((_, tg), _)| !tg)
        .map(|(_, c)| c.0)
        .sum();
    eprintln!(
        "[trace census: {total} dispatches over {steps} steps (whole run, prefill included), \
         {} kernels, {threads} dispatched by threads]",
        census.len()
    );
    for ((name, tg), (n, ordered)) in &census {
        eprintln!(
            "  {n:6} {} ordered={ordered:<6} {name}",
            if *tg { "tg " } else { "thr" }
        );
    }
    flow.print(steps);
}

/// A live value in [`OutputFlow`]: `(producer dispatch, producer name,
/// readers, first reader, last reader)`.
type Live = (u64, u32, u32, u64, u64);

/// The trace's outputs as values (`#814`, `DESIGN.md` §9.2aa): each output
/// binding at a `(buffer, offset)` is one value, live until the next output
/// binding there, and its readers are the later dispatches that bind it as
/// an input. An output read by exactly one dispatch, the one right after its
/// producer, is the fusion candidate: its only purpose is to hand a result to
/// the next kernel. Keyed on `(buffer, offset)`, so two bindings of one
/// region at different offsets are not matched; a binding a kernel both
/// reads and writes counts as a new value only.
#[cfg(feature = "metal")]
#[derive(Default)]
struct OutputFlow {
    names: Vec<String>,
    name_of: std::collections::HashMap<String, u32>,
    /// Per live value, keyed by `(buffer, offset)`.
    live: std::collections::HashMap<(u64, usize), Live>,
    at: u64,
    outputs: u64,
    unread: u64,
    next_only: u64,
    pairs: std::collections::HashMap<(u32, u32), u64>,
    /// Kernel of each dispatch, by trace position.
    kernel_of: Vec<u32>,
}

#[cfg(feature = "metal")]
impl OutputFlow {
    fn dispatch(&mut self, d: &lloom_tensor::kernels::metal::trace::Dispatch) {
        let i = self.at;
        self.at += 1;
        let id = match self.name_of.get(&d.pipeline) {
            Some(&id) => id,
            None => {
                let id = self.names.len() as u32;
                self.names.push(d.pipeline.clone());
                self.name_of.insert(d.pipeline.clone(), id);
                id
            }
        };
        self.kernel_of.push(id);
        for b in d.bindings.iter().filter(|b| !b.is_output) {
            if let Some(v) = self.live.get_mut(&(b.buffer_id, b.offset)) {
                if v.4 != i {
                    v.2 += 1;
                    if v.2 == 1 {
                        v.3 = i;
                    }
                    v.4 = i;
                }
            }
        }
        for b in d.bindings.iter().filter(|b| b.is_output) {
            let fresh = (i, id, 0u32, u64::MAX, u64::MAX);
            if let Some(old) = self.live.insert((b.buffer_id, b.offset), fresh) {
                self.retire(old);
            }
        }
    }

    fn retire(&mut self, (producer, name, readers, first, _): (u64, u32, u32, u64, u64)) {
        self.outputs += 1;
        match readers {
            0 => self.unread += 1,
            1 if first == producer + 1 => {
                self.next_only += 1;
                let consumer = self
                    .kernel_of
                    .get(first as usize)
                    .copied()
                    .unwrap_or(u32::MAX);
                *self.pairs.entry((name, consumer)).or_default() += 1;
            }
            _ => {}
        }
    }

    fn print(mut self, steps: u64) {
        let live: Vec<_> = self.live.drain().map(|(_, v)| v).collect();
        for v in live {
            self.retire(v);
        }
        eprintln!(
            "[trace outputs: {} outputs over {steps} steps (whole run); read only by the next \
             dispatch: {}; never read on the GPU: {}]",
            self.outputs, self.next_only, self.unread
        );
        let mut pairs: Vec<_> = self.pairs.into_iter().collect();
        pairs.sort_by_key(|p| std::cmp::Reverse(p.1));
        let name = |k: u32| self.names.get(k as usize).map_or("<none>", |s| s.as_str());
        for ((p, c), n) in pairs {
            eprintln!("  {n:8} {} -> {}", name(p), name(c));
        }
    }
}

/// Start recording every dispatch for [`print_trace_census`] when
/// `CANDLE_METAL_TRACE` asks for it; off, a dispatch pays one `OnceLock` read.
pub fn arm_trace() {
    #[cfg(feature = "metal")]
    if lloom_tensor::kernels::metal::trace::trace_requested() {
        lloom_tensor::kernels::metal::trace::set_recording(true);
    }
}

/// The command-buffer cut policy in force, read back from the device rather
/// than from the environment that selected it (§7.1a-i), with its period.
pub fn cut_policy_arm(device: &Device) -> String {
    #[cfg(feature = "metal")]
    if let Device::Metal(dev) = device {
        return lloom_tensor::metal_backend::CutPolicy::render(
            dev.cut_policy(),
            dev.compute_per_buffer(),
        );
    }
    let _ = device;
    "n/a".to_string()
}

/// The period a pipelined step loop raises the cut to; `cut` names the one
/// the prefill and a blocking loop run at.
pub fn pipelined_cut_arm(device: &Device) -> String {
    #[cfg(feature = "metal")]
    if let Device::Metal(dev) = device {
        return dev.pipelined_compute_per_buffer().to_string();
    }
    let _ = device;
    "n/a".to_string()
}

/// The param style the decode path ran, read from the process-wide default
/// rather than from the flag that set it: `packed` under `--packed` or
/// `--replay`, `split` (inline `setBytes`) otherwise.
#[cfg(feature = "metal")]
pub fn param_style_arm() -> &'static str {
    use lloom_tensor::kernels::ParamStyle;
    match lloom_tensor::kernels::default_param_style() {
        ParamStyle::Packed => "packed",
        ParamStyle::Split => "split",
    }
}

/// Off Metal every dispatch binds inline.
#[cfg(not(feature = "metal"))]
pub fn param_style_arm() -> &'static str {
    "split"
}

/// The GEMM tile a prefill forward of `m` tokens selects, for the spark line:
/// `m == 1` is the GEMV path, `m <= 32` the 16-row tile (aligned iff
/// `m % 16 == 0`), wider the 64-row tile (aligned iff `m % 64 == 0`).
pub fn tile_tag(m: usize) -> &'static str {
    const SKINNY_MAX: usize = 32;
    if m == 1 {
        "seq1"
    } else if m <= SKINNY_MAX {
        if m.is_multiple_of(16) {
            "16row-aligned"
        } else {
            "16row-UNALIGNED"
        }
    } else if m.is_multiple_of(64) {
        "64row-aligned"
    } else {
        "64row-UNALIGNED"
    }
}

/// The process's (minor, major) page-fault counts so far; zeros where
/// `getrusage` is unavailable.
fn rusage_faults() -> (u64, u64) {
    #[cfg(unix)]
    {
        // SAFETY: `rusage` is plain data, zeroed is a valid value, and the
        // pointer is to that live local.
        let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
        if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) } == 0 {
            return (ru.ru_minflt as u64, ru.ru_majflt as u64);
        }
    }
    (0, 0)
}

/// The `p`-quantile of an ascending `v` by nearest rank; NaN when empty.
pub fn quantile(v: &[f64], p: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v[(((v.len() - 1) as f64) * p).round() as usize]
}

/// The gaps between consecutive stamps (ms since a common start), as the
/// visible-cadence line reads them: how bursty the stream is at the consumer.
pub fn print_visible_cadence(stamps_ms: &[f64], tokens: usize) {
    if stamps_ms.len() < 2 {
        return;
    }
    let mut gaps: Vec<f64> = stamps_ms.windows(2).map(|w| w[1] - w[0]).collect();
    gaps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q = |f: f64| gaps[((gaps.len() - 1) as f64 * f) as usize];
    let burst_1ms = gaps.iter().filter(|g| **g < 1.0).count();
    eprintln!(
        "[visible cadence: {} writes over {} tokens; gap p50 {:.2} / p90 {:.2} / p99 {:.2} / max {:.2} ms; {} gaps < 1 ms (tokens arriving in a burst)]",
        stamps_ms.len(),
        tokens,
        q(0.5),
        q(0.9),
        q(0.99),
        gaps[gaps.len() - 1],
        burst_1ms
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_tile_tags_follow_the_gemm_tile_model() {
        assert_eq!(tile_tag(1), "seq1");
        assert_eq!(tile_tag(16), "16row-aligned");
        assert_eq!(tile_tag(32), "16row-aligned");
        assert_eq!(tile_tag(8), "16row-UNALIGNED");
        assert_eq!(tile_tag(64), "64row-aligned");
        assert_eq!(tile_tag(100), "64row-UNALIGNED");
    }

    #[test]
    fn the_quantile_is_nearest_rank() {
        let v = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(quantile(&v, 0.0), 1.0);
        assert_eq!(quantile(&v, 0.5), 3.0);
        assert_eq!(quantile(&v, 1.0), 5.0);
        assert!(quantile(&[], 0.5).is_nan());
    }
}
