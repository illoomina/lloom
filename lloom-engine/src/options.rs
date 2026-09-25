//! The engine's option groups: what a run selects, as structs
//! (`#[derive(Options)]` — the flags, the dotted keys, the JSON, the env
//! aliases and the `knobs` table are generated from these field lists).
//!
//! Nothing here runs per token. A group is resolved once — at load
//! ([`ReadbackConfig::arm`]) or at the request ([`DecodeConfig::options`]) —
//! into the library types the loop already reads (`ReadbackArm`,
//! `DecodeOptions`), so the option surface adds no work to the token path
//! (`DESIGN.md` §15.2 #9).
//!
//! The groups mirror what `lfm2-chat` reaches by hand (`decode-driver-design.md`
//! Stage 7): the token mechanisms (`decode.readback.*`), the loop's pacing
//! (`decode.in_flight`), and the prefill's shape (`decode.prefill.*`). The
//! spellings are `lfm2-chat`'s, so a measurement written against it runs
//! unchanged through `lloom chat`.

use crate::decode::{ReadbackArm, RingArm};
use lloom::generation::poll::PollMechanism;
use lloom::generation::{CommitBudget, EosRead, WAKE_LIVE_EVERY_DEFAULT};
use lloom::nn::{DecodeOptions, PassWidths};
use lloom_macros::Options;
use serde::{Deserialize, Serialize};

// ---- the readback arm --------------------------------------------------------

/// Which readback strategy a run takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReadbackKind {
    /// The ring where the sampler admits it (argmax, plain temperature,
    /// gumbel), the blocking arm otherwise — resolved per request.
    #[default]
    Auto,
    /// The host reads every token back; any sampler, no lookahead.
    Blocking,
    /// The device-resident token ring; EOS learned per [`EosKind`].
    Ring,
}

impl std::fmt::Display for ReadbackKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Auto => "auto",
            Self::Blocking => "blocking",
            Self::Ring => "ring",
        })
    }
}

impl std::str::FromStr for ReadbackKind {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "auto" => Ok(Self::Auto),
            "blocking" => Ok(Self::Blocking),
            "ring" => Ok(Self::Ring),
            other => Err(format!("`{other}` is not a readback kind (auto, blocking, ring)")),
        }
    }
}

lloom_tensor::serde_via_str!(ReadbackKind);

/// How a ring learns EOS (`lloom::generation::EosRead`, by name).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EosKind {
    /// One blocking flag read per step (`lfm2-chat --readback ring`).
    Blocking,
    /// The kernel-written stamp polled (`ring-nonblocking`, the crown).
    #[default]
    Poll,
    /// No device EOS; the host scans the drained ids on the commit cadence
    /// (`ring-hostdrain`).
    HostDrain,
    /// One `eos_index` dispatch per commit window, read with a tail wait
    /// (`ring-gpubatch`).
    GpuBatched,
    /// The same index staged ahead of the lookahead forward, read once
    /// complete (`ring-gpubatch-nonblocking`).
    GpuBatchedNonblocking,
}

impl EosKind {
    /// The `EosRead` this kind names, the poll under `poll`.
    pub fn eos_read(self, poll: PollMechanism) -> EosRead {
        match self {
            Self::Blocking => EosRead::Blocking,
            Self::Poll => EosRead::Poll(poll),
            Self::HostDrain => EosRead::HostOnDrain,
            Self::GpuBatched => EosRead::GpuBatched,
            Self::GpuBatchedNonblocking => EosRead::GpuBatchedNonblocking,
        }
    }

    /// The kind an `EosRead` is, for a report line.
    pub fn of(eos: EosRead) -> Self {
        match eos {
            EosRead::Blocking => Self::Blocking,
            EosRead::Poll(_) => Self::Poll,
            EosRead::HostOnDrain => Self::HostDrain,
            EosRead::GpuBatched => Self::GpuBatched,
            EosRead::GpuBatchedNonblocking => Self::GpuBatchedNonblocking,
        }
    }
}

impl std::fmt::Display for EosKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Blocking => "blocking",
            Self::Poll => "poll",
            Self::HostDrain => "host-drain",
            Self::GpuBatched => "gpu-batched",
            Self::GpuBatchedNonblocking => "gpu-batched-nonblocking",
        })
    }
}

impl std::str::FromStr for EosKind {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "blocking" => Ok(Self::Blocking),
            "poll" | "nonblocking" => Ok(Self::Poll),
            "host-drain" | "hostdrain" => Ok(Self::HostDrain),
            "gpu-batched" | "gpubatch" => Ok(Self::GpuBatched),
            "gpu-batched-nonblocking" | "gpubatch-nonblocking" => Ok(Self::GpuBatchedNonblocking),
            other => Err(format!(
                "`{other}` is not an EOS read (blocking, poll, host-drain, gpu-batched, gpu-batched-nonblocking)"
            )),
        }
    }
}

lloom_tensor::serde_via_str!(EosKind);

/// The token mechanisms: how the sampled token feeds the next step and how
/// EOS is learned (`decode-driver-design.md` seam A; `.dev/serving-catalogue.md`
/// §3.A). `--readback <spelling>` is the shorthand for `kind` + `eos` (+ the
/// lookahead and the commit for the named presets); the fields are the keys.
#[derive(Options, Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(default)]
pub struct ReadbackConfig {
    /// The strategy: `auto` (the ring where the sampler admits it), `blocking`, `ring`.
    #[opt(variants = ["auto", "blocking", "ring"])]
    pub kind: ReadbackKind,
    /// How a ring learns EOS.
    #[opt(variants = ["blocking", "poll", "host-drain", "gpu-batched", "gpu-batched-nonblocking"])]
    pub eos: EosKind,
    /// The poll mechanism under `eos = poll` (`wfe` on aarch64 by default).
    #[opt(variants = ["spin", "spin-hint", "wfe", "sleep", "sleep:<µs>"])]
    pub poll: PollMechanism,
    /// Forward passes the host enqueues before it has learned the previous step's EOS (ring arms): 0 or 1.
    #[opt(variants = ["0", "1"])]
    pub lookahead: usize,
    /// How often the loop commits and prunes the in-flight queue: `per-token` or `steps:<N>`.
    #[opt(flag = "commit-budget", variants = ["per-token", "steps:<N>"])]
    pub commit_budget: CommitBudget,
    /// The adaptive commit (lloom #735): a buffer closes when the GPU starts the previous one, no cadence; needs a ring arm and lookahead 1, drains through the reader.
    #[opt(bool)]
    pub adaptive_commit: bool,
    /// Under the adaptive commit: the most steps one buffer holds before it closes regardless (0 = 8).
    #[opt]
    pub run_ahead: usize,
    /// Under the adaptive commit: the EOS gate — the kernels past an EOS return at once (lloom #735 step 4).
    #[opt(bool)]
    pub eos_gate: bool,
    /// Under the adaptive commit: publish with the fused argmax·eos·stamp kernel (off = the six-launch chain, the A/B arm).
    #[opt(bool)]
    pub fused_publish: bool,
    /// Stream tokens off a ring-reader thread per token instead of the loop's commit-cadence drain (ring arms; implied by the adaptive commit).
    #[opt(bool)]
    pub stream_reader: bool,
    /// Commits between live completion-marker writes on the deferred-marker arms (`host-drain`, `gpu-batched*`).
    #[opt(flag = "wake-cadence")]
    pub wake_cadence: usize,
}

impl Default for ReadbackConfig {
    /// The crown: the ring polled, one forward of lookahead, a commit every
    /// eight steps, the adaptive commit — `ReadbackArm::CROWN`, what `lloom
    /// serve` has run since #702.
    fn default() -> Self {
        ReadbackConfig {
            kind: ReadbackKind::Auto,
            eos: EosKind::Poll,
            poll: PollMechanism::default(),
            lookahead: 1,
            commit_budget: CommitBudget::steps(8).expect("8 > 0"),
            adaptive_commit: true,
            run_ahead: 0,
            eos_gate: false,
            fused_publish: true,
            stream_reader: false,
            wake_cadence: WAKE_LIVE_EVERY_DEFAULT,
        }
    }
}

impl ReadbackConfig {
    /// Every spelling `--readback` takes (`lfm2-chat`'s and `lloom serve`'s).
    pub const SPELLINGS: &'static [&'static str] = &[
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

    /// Apply a `--readback` spelling: the kind and the EOS read it names,
    /// and the lookahead / adaptive commit where the preset's name says so
    /// (`ring-fixed`: the cadence, no adaptive; `ring-la0`: no lookahead).
    /// The other fields keep their values, so `--readback ring-hostdrain
    /// --lookahead 0` composes in either order.
    pub fn apply_spelling(cfg: &mut Self, s: &str) -> Result<(), String> {
        match s {
            "auto" => cfg.kind = ReadbackKind::Auto,
            "blocking" => cfg.kind = ReadbackKind::Blocking,
            "ring" => {
                cfg.kind = ReadbackKind::Ring;
                cfg.eos = EosKind::Blocking;
            }
            "ring-nonblocking" => {
                cfg.kind = ReadbackKind::Ring;
                cfg.eos = EosKind::Poll;
            }
            "ring-fixed" => {
                cfg.kind = ReadbackKind::Ring;
                cfg.eos = EosKind::Poll;
                cfg.lookahead = 1;
                cfg.adaptive_commit = false;
            }
            "ring-la0" => {
                cfg.kind = ReadbackKind::Ring;
                cfg.eos = EosKind::Poll;
                cfg.lookahead = 0;
                cfg.adaptive_commit = false;
            }
            "ring-hostdrain" => {
                cfg.kind = ReadbackKind::Ring;
                cfg.eos = EosKind::HostDrain;
            }
            "ring-gpubatch" => {
                cfg.kind = ReadbackKind::Ring;
                cfg.eos = EosKind::GpuBatched;
            }
            "ring-gpubatch-nonblocking" => {
                cfg.kind = ReadbackKind::Ring;
                cfg.eos = EosKind::GpuBatchedNonblocking;
            }
            other => {
                return Err(format!(
                    "`{other}` is not a readback spelling ({})",
                    Self::SPELLINGS.join(", ")
                ))
            }
        }
        Ok(())
    }

    /// The ring this group describes, whatever `kind` says — what `Auto`
    /// resolves to when the sampler admits a ring.
    pub fn ring_arm(&self) -> RingArm {
        RingArm {
            lookahead: self.lookahead.min(1),
            commit_steps: self.commit_budget.period(),
            adaptive: self.adaptive_commit,
            eos: self.eos.eos_read(self.poll),
            wake: self.wake_cadence,
        }
    }

    /// The arm requests take: `Auto` (resolved per request against the
    /// sampler), the blocking arm, or this group's ring.
    pub fn arm(&self) -> ReadbackArm {
        match self.kind {
            ReadbackKind::Auto => ReadbackArm::Auto,
            ReadbackKind::Blocking => ReadbackArm::Blocking,
            ReadbackKind::Ring => ReadbackArm::Ring(self.ring_arm()),
        }
    }

    /// The `--readback` spelling this group's kind + EOS read amounts to.
    pub fn spelling(&self) -> &'static str {
        match (self.kind, self.eos) {
            (ReadbackKind::Auto, _) => "auto",
            (ReadbackKind::Blocking, _) => "blocking",
            (ReadbackKind::Ring, EosKind::Blocking) => "ring",
            (ReadbackKind::Ring, EosKind::Poll) => "ring-nonblocking",
            (ReadbackKind::Ring, EosKind::HostDrain) => "ring-hostdrain",
            (ReadbackKind::Ring, EosKind::GpuBatched) => "ring-gpubatch",
            (ReadbackKind::Ring, EosKind::GpuBatchedNonblocking) => "ring-gpubatch-nonblocking",
        }
    }

    /// The refusals `lfm2-chat` makes before a run, made here once so every
    /// binary refuses the same combinations with the same reasons.
    pub fn check(&self) -> Result<(), String> {
        if self.eos_gate && !self.adaptive_commit {
            return Err("eos_gate needs adaptive_commit (`--eos-gate` needs `--adaptive-commit`)".into());
        }
        if self.lookahead > 1 {
            return Err(format!("lookahead {} is not 0 or 1", self.lookahead));
        }
        // `adaptive_commit` on the blocking arm is not an error: the arm
        // reads every token back and the cadence knobs do not apply to it
        // (the report line renders `blocking`, never a cadence, §7.1a-i).
        Ok(())
    }
}

// ---- the prefill ----------------------------------------------------------------

/// The aligned GEMM buckets `--spark` schedules through (`lfm2-chat`: `1` is
/// the `seq_len == 1` path, `16`/`32` the aligned 16-row tile, `64` the
/// aligned 64-row tile).
pub const SPARK_BUCKETS: [usize; 4] = [1, 16, 32, 64];

fn spark_bucket(n: usize) -> Result<usize, String> {
    if SPARK_BUCKETS.contains(&n) {
        Ok(n)
    } else {
        Err(format!(
            "spark bucket {n} is not aligned: it keys the unaligned 16-row GEMM variant \
             (m % 16 != 0, DESIGN §13.5o). Use one of {SPARK_BUCKETS:?}"
        ))
    }
}

/// Parse a `--spark` spelling into `(ramp, bulk)`: `RAMP[/tile:BULK]`, where
/// `RAMP` is `N` (one fixed first bucket) or `progressive:START,STOP` (the
/// buckets from `START` to `STOP`), and `/tile:BULK` chops the remainder into
/// `BULK`-wide forwards instead of a `seq_len == 1` tail. `lfm2-chat`'s
/// grammar, verbatim.
pub fn parse_spark(s: &str) -> Result<(Vec<usize>, usize), String> {
    let s = s.trim();
    let (ramp_str, bulk) = match s.split_once("/tile:") {
        Some((ramp, bulk_str)) => {
            let b: usize = bulk_str
                .trim()
                .parse()
                .map_err(|_| format!("spark /tile:BULK must be an integer, got {bulk_str:?}"))?;
            if b == 1 {
                return Err("spark /tile:1 is not a bulk tile (1 = seq_len 1, the tail); use 16, 32 or 64, or omit /tile".into());
            }
            (ramp, spark_bucket(b)?)
        }
        None => (s, 0),
    };
    let ramp_str = ramp_str.trim();
    let ramp = if let Some(range) = ramp_str.strip_prefix("progressive:") {
        let (a, b) = range.split_once(',').ok_or_else(|| {
            format!("spark progressive:START,STOP needs two buckets, e.g. progressive:1,64 (got {range:?})")
        })?;
        let start = spark_bucket(
            a.trim()
                .parse()
                .map_err(|_| format!("spark progressive START must be an integer, got {a:?}"))?,
        )?;
        let stop = spark_bucket(
            b.trim()
                .parse()
                .map_err(|_| format!("spark progressive STOP must be an integer, got {b:?}"))?,
        )?;
        if start > stop {
            return Err(format!("spark progressive:{start},{stop}: START must be <= STOP"));
        }
        SPARK_BUCKETS
            .iter()
            .copied()
            .filter(|&n| n >= start && n <= stop)
            .collect()
    } else {
        let n: usize = ramp_str.parse().map_err(|_| {
            format!(
                "spark takes a bucket ({SPARK_BUCKETS:?}) or progressive:START,STOP \
                 (optionally with /tile:BULK), got {ramp_str:?}"
            )
        })?;
        vec![spark_bucket(n)?]
    };
    Ok((ramp, bulk))
}

/// The `spark` field's parser: the spelling is kept once it validates.
fn parse_spark_field(s: &str) -> Result<Option<String>, String> {
    if s.is_empty() || s.eq_ignore_ascii_case("none") {
        return Ok(None);
    }
    parse_spark(s)?;
    Ok(Some(s.to_string()))
}

/// The prefill's shape (`.dev/serving-catalogue.md` §3.D): the TILE that
/// bounds one forward, the spark / ubatch that streams the first token, the
/// CHUNK the batched scheduler interleaves, and the prefill's own pacer depth.
#[derive(Options, Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(default)]
pub struct PrefillConfig {
    /// The TILE: the most prompt tokens one prefill forward takes (llama.cpp's `n_ubatch`); 0 = the whole prompt in one forward.
    #[opt(flag = "prefill-tile")]
    pub tile: usize,
    /// The spark: how many prompt tokens the FIRST forward takes, the remainder in one more (0 = one-shot).
    #[opt(flag = "prefill-ubatch")]
    pub ubatch: usize,
    /// The CHUNK under `--batch N`: the most prompt tokens fed per batched step as extra ragged rows (0 = a whole prompt at its arrival).
    #[opt(flag = "prefill-chunk")]
    pub chunk: usize,
    /// An aligned-bucket schedule for the prefill forwards: `N` or `progressive:START,STOP`, optionally `/tile:BULK`.
    #[opt(parser = "parse_spark_field", variants = ["1", "16", "32", "64", "progressive:START,STOP", "…/tile:BULK"])]
    pub spark: Option<String>,
    /// How a batched step's prompt pieces are sized: `lru` or `ladder` (DESIGN §9.2aa I4).
    #[opt(flag = "pass-widths", variants = ["lru", "ladder"])]
    pub pass_widths: PassWidths,
    /// The pacer's in-flight bound during prefill (0 = decode's depth).
    #[opt(flag = "prefill-in-flight", env = "LLOOM_PREFILL_IN_FLIGHT")]
    pub in_flight: usize,
}

impl Default for PrefillConfig {
    fn default() -> Self {
        PrefillConfig {
            tile: 512,
            ubatch: 0,
            chunk: 0,
            spark: None,
            pass_widths: PassWidths::Lru,
            in_flight: 0,
        }
    }
}

impl PrefillConfig {
    /// The spark's `(schedule, bulk)`, if one is set.
    pub fn schedule(&self) -> Option<(Vec<usize>, usize)> {
        self.spark.as_deref().map(|s| parse_spark(s).expect("validated when set"))
    }
}

// ---- the group ----------------------------------------------------------------

/// The decode loop's options: the readback arm, the pacer, the prefill.
/// Resolved once per request into `DecodeOptions` ([`Self::options`]).
#[derive(Options, Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
#[serde(default)]
pub struct DecodeConfig {
    /// The token mechanisms; `--readback <spelling>` is the shorthand.
    #[opt(nested, flag = "readback", parser = "ReadbackConfig::apply_spelling")]
    pub readback: ReadbackConfig,
    /// How many forwards the host may run ahead of the GPU (the pacer's bound); 0 = the depth warmup measured.
    #[opt(flag = "in-flight", env = "LLOOM_IN_FLIGHT")]
    pub in_flight: usize,
    /// Diagnostic, perturbs what it measures: drain the GPU per phase of a speculative round.
    #[opt(bool)]
    pub phase_sync: bool,
    /// The prefill's shape.
    #[opt(nested)]
    pub prefill: PrefillConfig,
}

impl DecodeConfig {
    /// The `DecodeOptions` for one run of `max_tokens` tokens on `arm`. The
    /// blocking arm has no lookahead and commits per token (the host holds
    /// every token); `stream_via_reader` and `adaptive_commit` are set by the
    /// loop once it holds a reader, never here. `tile` overrides the group's
    /// prefill tile when non-zero (a request's own).
    pub fn options(&self, max_tokens: usize, arm: ReadbackArm, tile: usize) -> DecodeOptions {
        let (schedule, bulk) = match self.prefill.schedule() {
            Some((s, b)) => (Some(s), b),
            None => (None, 0),
        };
        let ring = match arm {
            ReadbackArm::Ring(r) => Some(r),
            _ => None,
        };
        DecodeOptions {
            max_tokens,
            budget: match ring {
                Some(r) => CommitBudget::steps(r.commit_steps.max(1)).expect("> 0"),
                None => CommitBudget::per_token(),
            },
            lookahead: ring.map(|r| r.lookahead).unwrap_or(0),
            sample_split: self.phase_sync,
            in_flight_depth: self.in_flight,
            prefill_ubatch: self.prefill.ubatch,
            prefill_schedule: schedule,
            prefill_bulk: bulk,
            prefill_pacer_depth: self.prefill.in_flight,
            stream_via_reader: false,
            adaptive_commit: false,
            run_ahead: self.readback.run_ahead,
            stop: None,
            eos_gate: self.readback.eos_gate,
            fused_publish: self.readback.fused_publish,
            prefill_tile: if tile > 0 { tile } else { self.prefill.tile },
            prefill_chunk: self.prefill.chunk,
            pass_widths: self.prefill.pass_widths,
        }
    }

    /// The refusals, once (`ReadbackConfig::check`).
    pub fn check(&self) -> Result<(), String> {
        self.readback.check()
    }
}

/// Every tuning knob of the two library crates, as one group: the kernels,
/// the command stream, the pool and the CPU side (`lloom_tensor::tuning`)
/// beside the model ports, the streamed tier, the drafter and the ring
/// (`lloom::tuning`). The dotted keys are `tuning.<sub>.<key>`; each leaf's
/// env alias is the `LLOOM_*` / `CANDLE_*` / `QWEN4EXP_*` variable it
/// replaces, so the scripts keep working. Installed into both crates before
/// the model loads ([`TuningOptions::install`]).
#[derive(Options, Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
#[serde(default)]
pub struct TuningOptions {
    /// The dense GEMM/GEMV tiles.
    #[opt(nested)]
    pub gemm: lloom_tensor::tuning::GemmTuning,
    /// The quantized matvec cores.
    #[opt(nested)]
    pub qmv: lloom_tensor::tuning::QmvTuning,
    /// The quantized GEMM at prefill.
    #[opt(nested)]
    pub qmm: lloom_tensor::tuning::QmmTuning,
    /// The routed-expert GEMM.
    #[opt(nested)]
    pub moe: lloom_tensor::tuning::MoeTuning,
    /// The DeltaNet step kernel.
    #[opt(nested)]
    pub gdn: lloom_tensor::tuning::GdnTuning,
    /// The Metal command stream and the device.
    #[opt(nested)]
    pub metal: lloom_tensor::tuning::MetalTuning,
    /// The buffer pool.
    #[opt(nested)]
    pub pool: lloom_tensor::tuning::PoolTuning,
    /// The instruments.
    #[opt(nested)]
    pub instrument: lloom_tensor::tuning::InstrumentTuning,
    /// The CPU side.
    #[opt(nested)]
    pub cpu: lloom_tensor::tuning::CpuTuning,
    /// LFM2's runtime toggles.
    #[opt(nested)]
    pub lfm2: lloom::tuning::Lfm2Tuning,
    /// The DSpark drafter.
    #[opt(nested)]
    pub dspark: lloom::tuning::DsparkTuning,
    /// The streamed expert tier.
    #[opt(nested)]
    pub stream: lloom::tuning::StreamTuning,
    /// Qwen4-exp's own arms.
    #[opt(nested)]
    pub qwen4exp: lloom::tuning::Qwen4ExpTuning,
    /// The token ring and the decode thread.
    #[opt(nested)]
    pub ring: lloom::tuning::RingTuning,
}

impl TuningOptions {
    /// The tensor crate's half.
    pub fn tensor(&self) -> lloom_tensor::tuning::Tuning {
        lloom_tensor::tuning::Tuning {
            gemm: self.gemm.clone(),
            qmv: self.qmv.clone(),
            qmm: self.qmm.clone(),
            moe: self.moe.clone(),
            gdn: self.gdn.clone(),
            metal: self.metal.clone(),
            pool: self.pool.clone(),
            instrument: self.instrument.clone(),
            cpu: self.cpu.clone(),
        }
    }

    /// The model crate's half.
    pub fn lloom(&self) -> lloom::tuning::Tuning {
        lloom::tuning::Tuning {
            lfm2: self.lfm2.clone(),
            dspark: self.dspark.clone(),
            stream: self.stream.clone(),
            qwen4exp: self.qwen4exp.clone(),
            ring: self.ring.clone(),
        }
    }

    /// Install both halves for the process. `Err` names the half that had
    /// already been read (a knob a `OnceLock` site took is fixed; the report
    /// line would otherwise lie about it).
    pub fn install(&self) -> Result<(), String> {
        lloom_tensor::tuning::install(self.tensor())
            .map_err(|_| "tuning: the tensor crate's knobs were already read before install".to_string())?;
        lloom::tuning::install(self.lloom())
            .map_err(|_| "tuning: the model crate's knobs were already read before install".to_string())
    }
}

/// Every option group the engine reads, as one tree: the dotted keys are
/// `<group>.<key>` (`decode.readback.eos`, `tuning.qmv.q6k_nr0`), the JSON
/// is one object per group, and `lloom config` prints the resolved tree back
/// in this shape.
#[derive(Options, Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
#[serde(default)]
pub struct EngineOptions {
    /// The decode loop: the readback arm, the pacer, the prefill.
    #[opt(nested)]
    pub decode: DecodeConfig,
    /// Every tuning knob of the kernels, the device, the pool, the model ports.
    #[opt(nested)]
    pub tuning: TuningOptions,
}

impl EngineOptions {
    /// The refusals of every group, once.
    pub fn check(&self) -> Result<(), String> {
        self.decode.check()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spellings_round_trip_through_the_group() {
        for s in ReadbackConfig::SPELLINGS {
            let mut cfg = ReadbackConfig::default();
            ReadbackConfig::apply_spelling(&mut cfg, s).unwrap();
            let expect = match *s {
                "ring-fixed" | "ring-la0" => "ring-nonblocking",
                other => other,
            };
            assert_eq!(cfg.spelling(), expect, "{s}");
        }
        let mut cfg = ReadbackConfig::default();
        ReadbackConfig::apply_spelling(&mut cfg, "ring-la0").unwrap();
        assert_eq!(cfg.lookahead, 0);
        assert!(!cfg.adaptive_commit);
        assert!(ReadbackConfig::apply_spelling(&mut cfg, "ring-x").is_err());
    }

    #[test]
    fn the_default_is_the_crown() {
        let cfg = DecodeConfig::default();
        assert_eq!(cfg.readback.arm(), ReadbackArm::Auto);
        assert_eq!(cfg.readback.ring_arm(), RingArm::CROWN);
        let opts = cfg.options(7, ReadbackArm::Ring(cfg.readback.ring_arm()), 0);
        assert_eq!(opts.lookahead, 1);
        assert_eq!(opts.budget.period(), 8);
        assert_eq!(opts.prefill_tile, 512);
        let blocking = cfg.options(7, ReadbackArm::Blocking, 1024);
        assert_eq!(blocking.lookahead, 0);
        assert_eq!(blocking.budget.period(), 1);
        assert_eq!(blocking.prefill_tile, 1024);
    }

    #[test]
    fn the_group_reads_every_front_end() {
        let mut cfg = DecodeConfig::default();
        cfg.set("readback", "ring-hostdrain").unwrap();
        cfg.set("readback.wake_cadence", "2").unwrap();
        cfg.set("prefill.spark", "progressive:1,64/tile:64").unwrap();
        cfg.set("in_flight", "3").unwrap();
        assert_eq!(cfg.readback.eos, EosKind::HostDrain);
        assert_eq!(cfg.readback.ring_arm().wake, 2);
        assert_eq!(cfg.prefill.schedule(), Some((vec![1, 16, 32, 64], 64)));
        assert!(cfg.set("prefill.spark", "8").is_err());
        cfg.merge_json(&serde_json::json!({"readback": {"eos": "gpu-batched", "commit_budget": "steps:4"}}))
            .unwrap();
        assert_eq!(cfg.readback.eos, EosKind::GpuBatched);
        assert_eq!(cfg.readback.commit_budget.period(), 4);
        cfg.merge_json(&serde_json::json!({"readback": "blocking"})).unwrap();
        assert_eq!(cfg.readback.kind, ReadbackKind::Blocking);
        let dumped = serde_json::to_value(&cfg).unwrap();
        let mut back = DecodeConfig::default();
        back.merge_json(&dumped).unwrap();
        assert_eq!(back, cfg);
        assert!(cfg.check().is_ok(), "the cadence knobs are ignored on the blocking arm, not refused");
        cfg.set("readback.eos_gate", "true").unwrap();
        cfg.set("readback.adaptive_commit", "false").unwrap();
        assert!(cfg.check().is_err(), "the EOS gate needs the adaptive commit");
    }

    #[test]
    fn the_flags_are_lfm2_chats() {
        use clap::Parser;
        #[derive(Parser)]
        struct Cli {
            #[command(flatten)]
            decode: DecodeConfigArgs,
        }
        let cli = Cli::parse_from([
            "t",
            "--readback",
            "ring-gpubatch-nonblocking",
            "--lookahead",
            "0",
            "--commit-budget",
            "per-token",
            "--adaptive-commit=false",
            "--wake-cadence",
            "1",
            "--in-flight",
            "2",
            "--prefill-tile",
            "2048",
            "--spark",
            "16",
        ]);
        let mut cfg = DecodeConfig::default();
        cli.decode.apply(&mut cfg).unwrap();
        assert_eq!(cfg.readback.eos, EosKind::GpuBatchedNonblocking);
        assert_eq!(cfg.readback.lookahead, 0);
        assert_eq!(cfg.readback.commit_budget.period(), 1);
        assert!(!cfg.readback.adaptive_commit);
        assert_eq!(cfg.readback.wake_cadence, 1);
        assert_eq!(cfg.in_flight, 2);
        assert_eq!(cfg.prefill.tile, 2048);
        assert_eq!(cfg.prefill.schedule(), Some((vec![16], 0)));
    }
}
