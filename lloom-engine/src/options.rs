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
//! The groups are the token mechanisms (`decode.readback.*`), the loop's
//! pacing (`decode.in_flight`), and the prefill's shape (`decode.prefill.*`).
//! The spellings are the retired `lfm2-chat` harness's, so a measurement
//! written against it runs through `lloom chat` / `lloom bench`; the
//! flag-for-flag map is `decode-driver-design.md` Stage 7.

use crate::decode::{DecodeParams, ReadbackArm, RingArm};
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
            other => Err(format!(
                "`{other}` is not a readback kind (auto, blocking, ring)"
            )),
        }
    }
}

lloom_tensor::serde_via_str!(ReadbackKind);

/// How a ring learns EOS (`lloom::generation::EosRead`, by name).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EosKind {
    /// One blocking flag read per step (`--readback ring`).
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
    #[opt(variants = ["auto", "blocking", "ring"], path = "decode.readback.kind", status = "advanced", facets = ["buffer", "locality"], flows = ["token-out"])]
    pub kind: ReadbackKind,
    /// How a ring learns EOS.
    #[opt(variants = ["blocking", "poll", "host-drain", "gpu-batched", "gpu-batched-nonblocking"], path = "decode.readback.eos", status = "arm", facets = ["signal"], flows = ["token-out"])]
    pub eos: EosKind,
    /// The poll mechanism under `eos = poll` (`wfe` on aarch64 by default).
    #[opt(variants = ["spin", "spin-hint", "wfe", "sleep", "sleep:<µs>"], path = "decode.readback.wait", status = "arm", facets = ["wait"], flows = ["token-out"], forms = ["spin", "spin-hint", "wfe", "sleep"], from_v1 = "lloom_tensor::options::v1::poll_from", to_v1 = "lloom_tensor::options::v1::poll_to", schema = "Wait", grammar = "lloom_tensor::options::wait::READBACK_WAIT")]
    pub poll: PollMechanism,
    /// Forward passes the host enqueues before it has learned the previous step's EOS (ring arms): 0 or 1.
    #[opt(variants = ["0", "1"], path = "decode.readback.lookahead_forwards", status = "advanced", facets = ["lead"], flows = ["token-out", "commands-out"])]
    pub lookahead: usize,
    /// How often the loop commits and prunes the in-flight queue: `per-token` or `steps:<N>`.
    #[opt(flag = "commit-budget", variants = ["per-token", "steps:<N>"], path = "decode.readback.commit_budget", status = "arm", facets = ["cadence"], flows = ["token-out", "commands-out"])]
    pub commit_budget: CommitBudget,
    /// The adaptive commit (lloom #735): a buffer closes when the GPU starts the previous one, no cadence; needs a ring arm and lookahead 1, drains through the reader.
    #[opt(bool, path = "decode.readback.adaptive_commit", status = "advanced", facets = ["cadence"], flows = ["token-out", "commands-out"])]
    pub adaptive_commit: bool,
    /// Under the adaptive commit: the most steps one buffer holds before it closes regardless (0 = 8).
    #[opt(path = "decode.readback.run_ahead_steps", status = "arm", facets = ["lead", "cadence"], flows = ["token-out"])]
    pub run_ahead: usize,
    /// The EOS gate: the kernels queued past an EOS return at once, so the cache ends at the EOS. Rides the adaptive commit; off where the adaptive commit does not run, or under the ICB replay executor.
    #[opt(bool, path = "decode.readback.eos_gate", status = "advanced", facets = ["lead"], flows = ["token-out", "commands-out"])]
    pub eos_gate: bool,
    /// Under the adaptive commit: publish with the fused argmax·eos·stamp kernel (off = the six-launch chain, the A/B arm).
    #[opt(bool, path = "decode.readback.fused_publish", status = "arm", facets = ["produce"], flows = ["token-out"])]
    pub fused_publish: bool,
    /// Stream tokens off a ring-reader thread per token instead of the loop's commit-cadence drain (ring arms; implied by the adaptive commit).
    #[opt(bool, path = "decode.readback.stream_reader", status = "arm", facets = ["consume"], flows = ["token-out"])]
    pub stream_reader: bool,
    /// Commits between live completion-marker writes on the deferred-marker arms (`host-drain`, `gpu-batched*`).
    #[opt(flag = "wake-cadence", path = "decode.readback.wake_every_commits", status = "arm", facets = ["signal", "cadence"], flows = ["token-out"])]
    pub wake_cadence: usize,
}

impl Default for ReadbackConfig {
    /// The flow: the ring polled, one forward of lookahead, the adaptive
    /// commit with its reader, and the EOS gate — `ReadbackArm::CROWN` plus
    /// the gate (`docs/dispatch.md` §11.5z-xiv).
    fn default() -> Self {
        ReadbackConfig {
            kind: ReadbackKind::Auto,
            eos: EosKind::Poll,
            poll: PollMechanism::default(),
            lookahead: 1,
            commit_budget: CommitBudget::steps(8).expect("8 > 0"),
            adaptive_commit: true,
            run_ahead: 0,
            eos_gate: true,
            fused_publish: true,
            stream_reader: false,
            wake_cadence: WAKE_LIVE_EVERY_DEFAULT,
        }
    }
}

impl ReadbackConfig {
    /// Every spelling `--readback` takes.
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
        let s = lloom_tensor::options::parse_arm(s, Self::SPELLINGS)
            .map_err(|reason| format!("readback: {reason}"))?;
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
    /// resolves to when the sampler admits a ring. It is the arm that runs:
    /// on the fixed cadence where the arm cannot carry the adaptive commit
    /// ([`RingArm::resolved`], [`Self::adaptive_clamp`]).
    pub fn ring_arm(&self) -> RingArm {
        self.requested_ring_arm().resolved()
    }

    /// Why the adaptive commit this group asks for does not run, or `None`
    /// when it runs or is not asked for (or the kind is `blocking`, whose
    /// arm has no cadence). The EOS gate rides the adaptive commit, so the
    /// note names it too when it is on. Printed at load beside the arm that
    /// runs.
    pub fn adaptive_clamp(&self) -> Option<String> {
        if !self.adaptive_commit || self.kind == ReadbackKind::Blocking {
            return None;
        }
        let ring = self.requested_ring_arm();
        ring.why_not_adaptive().map(|why| {
            format!(
                "adaptive_commit is off on {}: {why}; the ring runs on the fixed cadence{}",
                ring.resolved().name(),
                if self.eos_gate {
                    ", without the EOS gate that rides it"
                } else {
                    ""
                }
            )
        })
    }

    fn requested_ring_arm(&self) -> RingArm {
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

    /// The refusals a run meets before it starts, made here once so every
    /// binary refuses the same combinations with the same reasons. Only a
    /// value no arm can run is refused: a knob that defaults on (the adaptive
    /// commit, the EOS gate that rides it) is clamped where the arm cannot
    /// carry it ([`Self::adaptive_clamp`]), and the stream reader streams
    /// every ring arm (`docs/dispatch.md` §11.5z-xvi).
    pub fn check(&self) -> Result<(), String> {
        if self.lookahead > 1 {
            return Err(format!(
                "C10: decode.readback.lookahead_forwards = {} is not 0 or 1",
                self.lookahead
            ));
        }
        Ok(())
    }
}

/// Whether a ring arm, as it runs ([`RingArm::resolved`]), stamps each token
/// as it publishes it: the poll and flag reads always, the deferred reads
/// only under the adaptive commit. A model's host fill needs that stamp
/// (`CausalLM::host_feed`); on any other arm its forward reads the fed id
/// back. Restates `ReadbackStrategy::stamps_each_token` for `Ring`, and a
/// test holds the two equal on every read and cadence.
pub fn stamps_each_token(ring: RingArm) -> bool {
    matches!(ring.eos, EosRead::Blocking | EosRead::Poll(_)) || ring.adaptive
}

// ---- the prefill ----------------------------------------------------------------

/// The aligned GEMM buckets `--spark` schedules through: `1` is the
/// `seq_len == 1` path, `16`/`32` the aligned 16-row tile, `64` the aligned
/// 64-row tile.
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
/// `BULK`-wide forwards instead of a `seq_len == 1` tail.
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
            return Err(format!(
                "spark progressive:{start},{stop}: START must be <= STOP"
            ));
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
    #[opt(flag = "prefill-tile", path = "prefill.tile", status = "advanced", cli = "prefill-tile", facets = ["cadence"], flows = ["prompt-in"])]
    pub tile: usize,
    /// The spark: how many prompt tokens the FIRST forward takes, the remainder in one more (0 = one-shot).
    #[opt(flag = "prefill-ubatch", path = "prefill.first_forward", status = "arm", facets = ["cadence"], flows = ["prompt-in"])]
    pub ubatch: usize,
    /// The CHUNK under `--batch N`: the most prompt tokens fed per batched step as extra ragged rows (0 = a whole prompt at its arrival).
    #[opt(flag = "prefill-chunk", path = "scheduler.prefill_chunk", status = "advanced", cli = "prefill-chunk", facets = ["cadence"], flows = ["prompt-in"])]
    pub chunk: usize,
    /// An aligned-bucket schedule for the prefill forwards: `N` or `progressive:START,STOP`, optionally `/tile:BULK`.
    #[opt(parser = "parse_spark_field", variants = ["1", "16", "32", "64", "progressive:START,STOP", "…/tile:BULK"], path = "prefill.schedule", status = "arm", facets = ["cadence"], flows = ["prompt-in"])]
    pub spark: Option<String>,
    /// How a batched step's prompt pieces are sized: `lru` or `ladder` (DESIGN §9.2aa I4).
    #[opt(flag = "pass-widths", variants = ["lru", "ladder"], path = "scheduler.pass_widths", status = "arm")]
    pub pass_widths: PassWidths,
    /// The pacer's in-flight bound during prefill (0 = decode's depth).
    #[opt(flag = "prefill-in-flight", env = "LLOOM_PREFILL_IN_FLIGHT", path = "prefill.in_flight_forwards", status = "advanced", facets = ["lead"], flows = ["prompt-in", "commands-out"])]
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
        self.spark
            .as_deref()
            .map(|s| parse_spark(s).expect("validated when set"))
    }
}

// ---- the group ----------------------------------------------------------------

/// The decode loop's options: the readback arm, the pacer, the prefill.
/// Resolved once per request into `DecodeOptions` ([`Self::options`]).
#[derive(Options, Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
#[serde(default)]
pub struct DecodeConfig {
    /// The readback arm by name — `auto`, `blocking`, `ring`, `ring-nonblocking`, `ring-fixed`, `ring-la0`, `ring-hostdrain`, `ring-gpubatch`, `ring-gpubatch-nonblocking` — a shorthand for the `decode.readback` keys it names.
    #[opt(
        nested,
        flag = "readback",
        parser = "ReadbackConfig::apply_spelling",
        path = "decode.readback",
        status = "advanced",
        cli = "readback"
    )]
    pub readback: ReadbackConfig,
    /// How many forwards the host may run ahead of the GPU (the pacer's bound); 0 = the depth warmup measured.
    #[opt(flag = "in-flight", env = "LLOOM_IN_FLIGHT", path = "decode.in_flight_forwards", status = "advanced", facets = ["lead"], flows = ["token-out", "commands-out"])]
    pub in_flight: usize,
    /// Diagnostic, perturbs what it measures: drain the GPU per phase of a speculative round.
    #[opt(bool, path = "diagnostics.phase_sync", status = "instrument", cli = "phase-sync", facets = ["wait"], flows = ["commands-out"])]
    pub phase_sync: bool,
    /// Diagnostic, perturbs what it measures: hash every decode step's logits (each value's f32 bits, little-endian, one SHA-256 over the run — the harnesses' `logits_sha256`) and report it beside the tokens' digest; runs the blocking readback, and under the speculative body hashes the rows each round keeps.
    #[opt(bool, path = "diagnostics.logits_digest", status = "instrument", cli = "logits-digest", facets = ["consume"], flows = ["token-out"])]
    pub logits_digest: bool,
    /// The prefill's shape.
    #[opt(nested)]
    pub prefill: PrefillConfig,
}

impl DecodeConfig {
    /// The `DecodeOptions` for one run of `max_tokens` tokens on `arm`. The
    /// blocking arm has no lookahead and commits per token (the host holds
    /// every token); `stream_via_reader`, `adaptive_commit` and the EOS gate
    /// that rides it are set by the loop once it holds a reader, never here.
    /// `tile` overrides the group's prefill tile when non-zero (a request's
    /// own).
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
            host_feed_wait: None,
            eos_park: None,
            eos_watch: None,
            adaptive_commit: false,
            run_ahead: self.readback.run_ahead,
            stop: None,
            eos_gate: false,
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
/// `lloom chat|serve` ship the `fast` contract, the library's default
/// (RESTRUCTURING_DESIGN.md §2); a gate run through `lloom` passes
/// `--profile deterministic` (`architecture.md` §2.3.5b).
#[derive(Options, Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
#[serde(default)]
pub struct TuningOptions {
    /// The determinism contract (`--profile fast|deterministic`).
    #[opt(nested)]
    pub contract: lloom_tensor::tuning::ContractTuning,
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
    /// The dense weight-group tier.
    #[opt(nested)]
    pub wg: lloom::tuning::WgTuning,
}

impl TuningOptions {
    /// The tensor crate's half.
    pub fn tensor(&self) -> lloom_tensor::tuning::Tuning {
        lloom_tensor::tuning::Tuning {
            contract: self.contract.clone(),
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
            wg: self.wg.clone(),
        }
    }

    /// Install both halves for the process. `Err` names the half that had
    /// already been read (a knob a `OnceLock` site took is fixed; the report
    /// line would otherwise lie about it).
    pub fn install(&self) -> Result<(), String> {
        lloom_tensor::tuning::install(self.tensor()).map_err(|_| {
            "tuning: the tensor crate's knobs were already read before install".to_string()
        })?;
        lloom::tuning::install(self.lloom()).map_err(|_| {
            "tuning: the model crate's knobs were already read before install".to_string()
        })
    }
}

// ---- the model axes -----------------------------------------------------------

/// The cache presets, plus `auto` (the engine's rule: the pool arms at
/// `--batch N`, the rewindable arms under `--dspark`, the crown otherwise).
/// `rewindable` without a drafter and `rewindable-notap` run the speculative
/// body's cache at K = 1, so its cost separates from the verify loop's.
pub const CACHE_PRESETS: &[&str] = &[
    "auto",
    "crown",
    "rewindable",
    "rewindable-notap",
    "pool",
    "pool-spec",
];

/// The model's own axes: a cache preset as the base, then per-axis overrides
/// by flag name (`attn`, `kv-append`, `conv-state`, `qkv-layout`,
/// `gemv-silu`, …) — the `#[axis]` fields of `lfm2::Config`, whose parsers
/// validate the values this group stores. A family without an axis refuses
/// it at load (the axes are LFM2's until Qwen4-exp's `Config` carries the
/// derive; `kv-quant` is every family's KV store).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(default)]
pub struct ModelConfig {
    /// The base preset: `auto | crown | rewindable | rewindable-notap | pool | pool-spec`.
    pub cache: String,
    /// Per-axis overrides, flag → value in the flag spelling, applied over
    /// the preset in key order.
    pub axes: std::collections::BTreeMap<String, String>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        ModelConfig {
            cache: "auto".into(),
            axes: Default::default(),
        }
    }
}

impl ModelConfig {
    /// The preset and the axes, as keys.
    pub const META: &'static [lloom_tensor::options::KnobMeta] = &[
        lloom_tensor::options::KnobMeta {
            key: "cache",
            flag: "cache",
            doc: "The LFM2 cache preset the axes are applied over: `auto` (the engine's rule: the pool arms at a batch, the rewindable arms under a drafter, the crown otherwise), `crown`, `rewindable`, `rewindable-notap`, `pool`, `pool-spec`.",
            variants: &["auto", "crown", "rewindable", "rewindable-notap", "pool", "pool-spec"],
            ty: "String",
            path: "models.lfm2.preset",
            status: lloom_tensor::options::Status::Advanced,
            ..lloom_tensor::options::KnobMeta::BASE
        },
        lloom_tensor::options::KnobMeta {
            key: "axes",
            nested: Some(lloom::models::lfm2::Config::AXES_META),
            ..lloom_tensor::options::KnobMeta::BASE
        },
    ];

    /// `cache`, or an axis by its flag name (`attn` or `axes.attn`),
    /// validated against `lfm2::Config`'s parser for that axis.
    pub fn set(&mut self, key: &str, value: &str) -> Result<(), String> {
        if key == "cache" {
            let v = value.trim().to_ascii_lowercase();
            if !CACHE_PRESETS.contains(&v.as_str()) {
                return Err(format!(
                    "cache: `{value}` is not one of {}",
                    CACHE_PRESETS.join(", ")
                ));
            }
            self.cache = v;
            return Ok(());
        }
        let key = key.strip_prefix("axes.").unwrap_or(key);
        if lloom_tensor::options::is_none(value) {
            self.axes.remove(key);
            return Ok(());
        }
        // Validate on a scratch config so a bad value is refused here, with
        // the axis's own message, rather than at load; keep the spelling the
        // axis parses (a bool as `true`/`false`, an arm in lower case).
        let is_bool = lloom::models::lfm2::Config::AXES_META
            .iter()
            .any(|a| a.key == key && a.is_bool);
        let value = if is_bool {
            lloom_tensor::options::parse_bool(value)
                .map_err(|e| format!("--{key}: {e}"))?
                .to_string()
        } else {
            lloom_tensor::options::non_empty(value)
                .map_err(|e| format!("--{key}: {e}"))?
                .to_ascii_lowercase()
        };
        let mut probe = lloom::models::lfm2::Config::default();
        probe.set_axis(key, &value)?;
        self.axes.insert(key.to_string(), value);
        Ok(())
    }

    pub fn merge_json(&mut self, json: &serde_json::Value) -> Result<(), String> {
        let Some(map) = json.as_object() else {
            return Err("model: expected a JSON object".into());
        };
        for (k, v) in map {
            match (k.as_str(), v) {
                ("axes", serde_json::Value::Object(axes)) => {
                    for (ak, av) in axes {
                        self.set(ak, &json_scalar(ak, av)?)?;
                    }
                }
                ("axes", other) => {
                    return Err(format!("model.axes: expected an object, got {other}"))
                }
                (k, v) => self.set(k, &json_scalar(k, v)?)?,
            }
        }
        Ok(())
    }

    /// The process environment ([`Self::from_env_with`]).
    pub fn from_env(&mut self) -> Result<(), String> {
        self.from_env_with(&|name: &str| {
            std::env::var_os(name).map(|v| v.to_string_lossy().into_owned())
        })
    }

    pub fn from_env_with(&mut self, env: &dyn Fn(&str) -> Option<String>) -> Result<(), String> {
        self.env_legacy(env)?;
        self.env_canonical(env)
    }

    /// The axes never had env names.
    #[doc(hidden)]
    pub fn env_legacy(&mut self, _env: &dyn Fn(&str) -> Option<String>) -> Result<(), String> {
        Ok(())
    }

    /// Each key's `LLOOM_<KEY>`.
    #[doc(hidden)]
    pub fn env_canonical(&mut self, env: &dyn Fn(&str) -> Option<String>) -> Result<(), String> {
        let preset = lloom_tensor::options::canonical_env(Self::META[0].path);
        if let Some(v) = env(&preset) {
            self.set("cache", &v)
                .map_err(|e| format!("{preset}: {e}"))?;
        }
        for a in lloom::models::lfm2::Config::AXES_META {
            let var = lloom_tensor::options::canonical_env(a.path);
            if let Some(v) = env(&var) {
                self.set(a.key, &v).map_err(|e| format!("{var}: {e}"))?;
            }
        }
        Ok(())
    }

    /// Apply the overrides onto a config built from the checkpoint and the
    /// preset.
    pub fn apply_axes(&self, config: &mut lloom::models::lfm2::Config) -> Result<(), String> {
        for (k, v) in &self.axes {
            config.set_axis(k, v)?;
        }
        Ok(())
    }
}

fn json_scalar(key: &str, v: &serde_json::Value) -> Result<String, String> {
    match v {
        serde_json::Value::String(s) => Ok(s.clone()),
        serde_json::Value::Bool(b) => Ok(b.to_string()),
        serde_json::Value::Number(n) => Ok(n.to_string()),
        serde_json::Value::Null => Ok("none".into()),
        other => Err(format!("model.{key}: expected a string, got {other}")),
    }
}

/// The flags of the model group: `--cache` and every `#[axis]` flag of
/// `lfm2::Config` (`--attn`, `--kv-dtype`, …), generated by its derive with
/// each axis's key.
#[derive(clap::Args, Clone, Debug, Default)]
pub struct ModelConfigArgs {
    /// The LFM2 cache preset the axes are applied over (`auto | crown | rewindable | rewindable-notap | pool | pool-spec`). [key: models.lfm2.preset]
    #[arg(
        id = "ModelConfig.cache",
        long = "cache",
        value_name = "PRESET",
        hide = true,
        help_heading = lloom_tensor::options::heading("models.lfm2.preset"),
        overrides_with = "ModelConfig.cache"
    )]
    pub cache: Option<String>,
    #[command(flatten)]
    pub axes: lloom::models::lfm2::ConfigArgs,
}

impl ModelConfigArgs {
    pub fn apply(&self, base: &mut ModelConfig) -> Result<(), String> {
        if let Some(c) = &self.cache {
            base.set("cache", c)?;
        }
        for (flag, value) in self.axes.overrides() {
            base.set(flag, &value)?;
        }
        Ok(())
    }

    /// The storage paths whose flags were given (the derive's `given`).
    pub fn given(&self, prefix: &str, out: &mut std::collections::BTreeSet<String>) {
        let at = |k: &str| {
            if prefix.is_empty() {
                k.to_string()
            } else {
                format!("{prefix}.{k}")
            }
        };
        if self.cache.is_some() {
            out.insert(at("cache"));
        }
        for (flag, _) in self.axes.overrides() {
            out.insert(at(&format!("axes.{flag}")));
        }
    }
}

// ---- sampling -------------------------------------------------------------------

fn parse_stop(v: &str) -> Result<Vec<String>, String> {
    Ok(v.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect())
}
/// `id:bias[,id:bias…]` — OpenAI's `logit_bias` map as a flag.
fn parse_logit_bias(v: &str) -> Result<std::collections::BTreeMap<u32, f64>, String> {
    let mut out = std::collections::BTreeMap::new();
    for pair in v.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let (id, bias) = pair
            .split_once(':')
            .ok_or_else(|| format!("`{pair}` is not `token_id:bias`"))?;
        let id: u32 = id
            .trim()
            .parse()
            .map_err(|_| format!("`{id}` in `{pair}` is not a token id"))?;
        let bias: f64 = bias
            .trim()
            .parse()
            .map_err(|_| format!("`{bias}` in `{pair}` is not a number"))?;
        out.insert(id, bias);
    }
    Ok(out)
}
fn parse_prompt_budget_mode(v: &str) -> Result<String, String> {
    match v.trim().to_ascii_lowercase().as_str() {
        "refuse" => Ok("refuse".into()),
        "truncate" => Ok("truncate".into()),
        other => Err(format!("`{other}` is not `refuse` or `truncate`")),
    }
}
fn parse_defaults(v: &str) -> Result<String, String> {
    match v.trim().to_ascii_lowercase().as_str() {
        "greedy" => Ok("greedy".into()),
        "checkpoint" => Ok("checkpoint".into()),
        other => Err(format!("`{other}` is not `greedy` or `checkpoint`")),
    }
}

/// The sampling defaults a run (a chat turn, a request that names none) is
/// decoded with — transformers' `GenerationConfig` subset the sampler
/// implements (`Sampling`: argmax, temperature, top-p, top-k, top-k-then-p,
/// gumbel-max). `none` on a field means the default source's value: greedy
/// (`defaults = greedy`), or the checkpoint's `generation_config.json`
/// (`defaults = checkpoint`, vLLM's `--generation-config`). A request's own
/// fields override per request.
#[derive(Options, Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(default)]
pub struct SamplingConfig {
    /// Where an unset field's value comes from: `greedy` (argmax; every recorded digest) or `checkpoint` (its `generation_config.json`).
    #[opt(flag = "sampling-defaults", parser = "parse_defaults", variants = ["greedy", "checkpoint"], path = "generation.defaults_from", status = "stable", cli = "sampling-defaults")]
    pub defaults: String,
    /// The temperature; 0 (or none under `greedy`) is argmax.
    #[opt(
        flag = "temperature",
        short = 't',
        path = "generation.temperature",
        status = "stable",
        cli = "temperature"
    )]
    pub temperature: Option<f64>,
    /// Nucleus sampling's p (top-p); needs a temperature.
    #[opt(
        flag = "top-p",
        path = "generation.top_p",
        status = "stable",
        cli = "top-p"
    )]
    pub top_p: Option<f64>,
    /// Top-k sampling's k; with top-p, top-k then top-p. Host-side: the ring arm is refused, the blocking arm runs.
    #[opt(
        flag = "top-k",
        path = "generation.top_k",
        status = "stable",
        cli = "top-k"
    )]
    pub top_k: Option<usize>,
    /// How a temperature draws: `multinomial` (one uniform per position and the inverse CDF; the default) or `gumbel` (gumbel-max, one noise variate per vocabulary entry). Each is keyed by the seed and the position of the token it draws, so every readback arm and a speculative run draw the same tokens at one seed.
    #[opt(bool, path = "generation.sampling.method", values = ["gumbel=true", "multinomial=false"], variants = ["multinomial", "gumbel"], legacy_keys = ["generation.device_sampler"], status = "stable", facets = ["produce"], flows = ["token-out"])]
    pub gumbel: bool,
    /// The seed; none draws a fresh one per run (repeats explore), a value reproduces.
    #[opt(
        flag = "seed",
        path = "generation.seed",
        status = "stable",
        cli = "seed"
    )]
    pub seed: Option<u64>,
    /// New tokens per run at most (a request's own `max_tokens` overrides).
    #[opt(
        flag = "max-tokens",
        short = 'n',
        alias = "max-tokens-default",
        path = "generation.max_tokens",
        status = "stable",
        cli = "max-tokens"
    )]
    pub max_tokens: usize,
    /// Never stop on EOS: exactly `max_tokens` tokens (the EOS mechanism runs and never fires).
    #[opt(
        flag = "ignore-eos",
        bool,
        path = "generation.ignore_eos",
        status = "advanced",
        cli = "ignore-eos"
    )]
    pub ignore_eos: bool,
    /// Stop strings, comma-separated (a request's own `stop` overrides).
    #[opt(
        flag = "stop",
        parser = "parse_stop",
        path = "generation.stop",
        status = "stable",
        cli = "stop"
    )]
    pub stop: Vec<String>,
    /// Repetition penalty over prompt and output tokens (> 0; 1 is off): a positive logit divided by it, a negative one multiplied.
    #[opt(
        flag = "repetition-penalty",
        path = "generation.repetition_penalty",
        status = "stable",
        cli = "repetition-penalty"
    )]
    pub repetition_penalty: Option<f64>,
    /// Presence penalty (OpenAI, [-2, 2]; 0 is off): subtracted once from every token the output holds.
    #[opt(
        flag = "presence-penalty",
        path = "generation.presence_penalty",
        status = "stable",
        cli = "presence-penalty"
    )]
    pub presence_penalty: Option<f64>,
    /// Frequency penalty (OpenAI, [-2, 2]; 0 is off): subtracted per occurrence in the output.
    #[opt(
        flag = "frequency-penalty",
        path = "generation.frequency_penalty",
        status = "stable",
        cli = "frequency-penalty"
    )]
    pub frequency_penalty: Option<f64>,
    /// Min-p ([0, 1]; 0 is off): drops tokens below min_p times the top token's probability; sampled arms only.
    #[opt(
        flag = "min-p",
        path = "generation.min_p",
        status = "stable",
        cli = "min-p"
    )]
    pub min_p: Option<f64>,
    /// Logit bias, `token_id:bias,…` (each bias in [-100, 100]), added before the penalties.
    #[opt(
        flag = "logit-bias",
        parser = "parse_logit_bias",
        path = "generation.logit_bias",
        status = "advanced",
        cli = "logit-bias"
    )]
    pub logit_bias: std::collections::BTreeMap<u32, f64>,
    /// Tokens the model may generate inside the thinking block its prompt opens (vLLM's `thinking_token_budget`); then the template's end marker is fed and the answer follows. Refused where the template writes no block.
    #[opt(
        flag = "thinking-budget",
        path = "reasoning.budget",
        status = "stable",
        cli = "thinking-budget"
    )]
    pub thinking_budget: Option<usize>,
    /// Text fed before the end marker when the thinking budget is spent, so the cut reads naturally (default: the marker alone).
    #[opt(
        flag = "thinking-budget-message",
        path = "reasoning.budget_message",
        status = "advanced"
    )]
    pub thinking_budget_message: Option<String>,
    /// The most tokens a prompt may take (vLLM's `truncate_prompt_tokens`); a longer one is refused or truncated per `prompt_budget_mode`.
    #[opt(
        flag = "prompt-budget",
        path = "prompt.max_tokens",
        status = "advanced",
        cli = "prompt-budget"
    )]
    pub prompt_budget: Option<usize>,
    /// Over the prompt budget: `refuse` (the default: a clean error, the prompt is never silently shortened) or `truncate` (drop the oldest whole messages, the system message kept; a raw prompt keeps BOS and its last tokens).
    #[opt(flag = "prompt-budget-mode", parser = "parse_prompt_budget_mode", variants = ["refuse", "truncate"], path = "prompt.overflow", status = "advanced")]
    pub prompt_budget_mode: String,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        SamplingConfig {
            defaults: "greedy".into(),
            temperature: None,
            top_p: None,
            top_k: None,
            gumbel: false,
            seed: None,
            max_tokens: 512,
            ignore_eos: false,
            stop: Vec::new(),
            repetition_penalty: None,
            presence_penalty: None,
            frequency_penalty: None,
            min_p: None,
            logit_bias: Default::default(),
            thinking_budget: None,
            thinking_budget_message: None,
            prompt_budget: None,
            prompt_budget_mode: "refuse".into(),
        }
    }
}

impl SamplingConfig {
    /// Fill the unset fields from a checkpoint's `generation_config.json`
    /// when `defaults = checkpoint`; a no-op under `greedy`.
    pub fn apply_checkpoint(&mut self, gc: Option<&lloom::hub::GenerationConfig>) {
        if self.defaults != "checkpoint" {
            return;
        }
        let Some(gc) = gc else { return };
        if self.temperature.is_none() {
            self.temperature = gc.temperature;
        }
        if self.top_p.is_none() {
            self.top_p = gc.top_p;
        }
        if self.top_k.is_none() {
            self.top_k = gc.top_k;
        }
        if self.repetition_penalty.is_none() {
            self.repetition_penalty = gc.repetition_penalty;
        }
        if self.min_p.is_none() {
            self.min_p = gc.min_p;
        }
    }

    /// The logits processors these fields name; an unset field is neutral.
    pub fn processors(&self) -> lloom::generation::Processors {
        let neutral = lloom::generation::Processors::default();
        lloom::generation::Processors {
            repetition_penalty: self
                .repetition_penalty
                .unwrap_or(neutral.repetition_penalty),
            presence_penalty: self.presence_penalty.unwrap_or(neutral.presence_penalty),
            frequency_penalty: self.frequency_penalty.unwrap_or(neutral.frequency_penalty),
            min_p: self.min_p.unwrap_or(neutral.min_p),
            logit_bias: self
                .logit_bias
                .iter()
                .map(|(&id, &b)| (id, b as f32))
                .collect(),
        }
    }

    /// Refuse an out-of-range processor value, by its field's name. Every
    /// layer — flags, `--set`, `--generation-config`, the env — lands here,
    /// since the JSON layer deserializes without the flag parsers.
    pub fn check(&self) -> Result<(), String> {
        self.processors()
            .validate()
            .map_err(|e| format!("sampling: {e}"))?;
        if self.prompt_budget == Some(0) {
            return Err("sampling: prompt_budget must be at least 1".into());
        }
        parse_prompt_budget_mode(&self.prompt_budget_mode)
            .map(|_| ())
            .map_err(|e| format!("sampling: prompt_budget_mode: {e}"))
    }

    /// The seed for one run.
    pub fn seed_or_random(&self) -> u64 {
        self.seed.unwrap_or_else(rand::random::<u64>)
    }

    /// The sampler arm these fields name (`LogitsProcessor`'s vocabulary).
    pub fn sampling(&self) -> lloom::generation::Sampling {
        DecodeParams::sampling_of(self.temperature, self.top_p, self.top_k, self.gumbel)
    }

    /// The per-run parameters for `eos_ids` on `readback`, the prefill tile.
    pub fn params(
        &self,
        eos_ids: &[u32],
        readback: ReadbackArm,
        prefill_tile: usize,
    ) -> DecodeParams {
        DecodeParams {
            seed: self.seed_or_random(),
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            gumbel: self.gumbel,
            processors: self.processors(),
            max_tokens: self.max_tokens,
            eos_ids: if self.ignore_eos {
                vec![u32::MAX]
            } else {
                eos_ids.to_vec()
            },
            stop_strings: self.stop.clone(),
            readback,
            logits_digest: false,
            prefill_tile,
            thinking: None,
            turn_tail: 0,
            no_checkpoint: false,
            logprobs: None,
            cold: false,
        }
    }
}

// ---- speculation ----------------------------------------------------------------

pub(crate) fn parse_method(v: &str) -> Result<String, String> {
    match v.trim().to_ascii_lowercase().as_str() {
        "none" | "off" => Ok("none".into()),
        "dspark" => Ok("dspark".into()),
        "mtp" => Ok("mtp".into()),
        "dflash" => Ok("dflash".into()),
        "auto" => Ok("auto".into()),
        other => Err(format!(
            "`{other}` is not a speculation method (none, dspark, mtp, dflash, auto)"
        )),
    }
}

/// The speculative body: which drafter, how many tokens a round, where its
/// weights are — vLLM's `--speculative-config` shape. `dspark` is LFM2's
/// (a DSpark checkpoint drafts a block, the target verifies it in one pass;
/// `k` clamps the proposal, the drafter's block is 9); `mtp` is Qwen4-exp's
/// (the checkpoint's MTP head re-applied `k` times; `drafter` is the sidecar
/// GGUF `qwen4exp-mtp-quantize write` produces). Both byte-identical to
/// greedy. `dflash` is Qwen4-exp's block drafter (the DSpark family's
/// backbone over the target's tapped HC residuals; `drafter` its checkpoint
/// directory, `k` its tokens a round, 3 of its block of 7 by default). The
/// drafter's context KV is `tuning.dspark.ctx_kv`; the phase drain is
/// `decode.phase_sync`.
#[derive(Options, Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(default)]
pub struct SpeculativeConfig {
    /// `none`, `dspark` (LFM2), `mtp` (Qwen4-exp), `dflash` (Qwen4-exp), or `auto`: the family's own drafter when one is at hand (DSpark when `drafter` names one or the HF cache holds one; on Qwen4-exp MTP when `drafter` names the sidecar or `~/.cache/lloom/qwen4exp-mtp-q8.gguf` exists, else DFlash when `drafter` names its directory or the HF cache holds the PixelML drafter, at `--batch 1`), else none.
    #[opt(flag = "spec-method", parser = "parse_method", variants = ["none", "dspark", "mtp", "dflash", "auto"], path = "speculative.method", status = "stable", cli = "speculate")]
    pub method: String,
    /// Tokens the drafter proposes per round (none = the drafter's own: DSpark's block of 9, DFlash 3 of its block of 7, MTP 1).
    #[opt(flag = "spec-k", path = "speculative.num_speculative_tokens", status = "stable", cli = "spec-tokens", aliases = ["dspark-k", "mtp-k"])]
    pub k: Option<usize>,
    /// The drafter's weights: a DSpark checkpoint directory (none = the newest LFM2.5-2.6B-DSpark snapshot in the HF cache), a DFlash checkpoint directory (none = the newest PixelML/Qwen3.8-Flash-Next-NVFP4-DFlash snapshot), or the MTP sidecar GGUF (required).
    #[opt(flag = "drafter", path = "speculative.model", status = "stable", cli = "spec-model", aliases = ["draft-dir"])]
    pub drafter: Option<std::path::PathBuf>,
    /// Reserved: the drafter's own options where it is a checkpoint of the target's family (its own `models.<family>` keys, which today it shares with the target). Not built; any value is refused.
    #[opt(
        flag = "spec-draft-overrides",
        parser = "parse_reserved_overrides",
        path = "speculative.draft.overrides",
        status = "advanced"
    )]
    pub draft_overrides: Option<String>,
}

/// `speculative.draft.overrides` is a reserved key: the name is the
/// tree's, the mechanism is not built.
fn parse_reserved_overrides(v: &str) -> Result<Option<String>, String> {
    if v.is_empty() || v.eq_ignore_ascii_case("none") {
        return Ok(None);
    }
    Err("speculative.draft.overrides is reserved: a same-family drafter's own options are not built (it shares the target's)".into())
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        SpeculativeConfig {
            method: "none".into(),
            k: None,
            drafter: None,
            draft_overrides: None,
        }
    }
}

// ---- the executor ---------------------------------------------------------------

fn parse_arena_layout(v: &str) -> Result<String, String> {
    match v.trim().to_ascii_lowercase().as_str() {
        "packed" => Ok("packed".into()),
        "non-aliasing" | "nonaliasing" | "reference" => Ok("non-aliasing".into()),
        other => Err(format!(
            "`{other}` is not an arena layout (packed, non-aliasing)"
        )),
    }
}
fn parse_replay_barriers(v: &str) -> Result<String, String> {
    match v.trim().to_ascii_lowercase().as_str() {
        "always" => Ok("always".into()),
        "skip-replayed" => Ok("skip-replayed".into()),
        other => Err(format!(
            "`{other}` is not a replay-barriers arm (always, skip-replayed)"
        )),
    }
}
fn parse_barrier_scope(v: &str) -> Result<String, String> {
    match v.trim().to_ascii_lowercase().as_str() {
        "run-start" => Ok("run-start".into()),
        "since-barrier" => Ok("since-barrier".into()),
        other => Err(format!(
            "`{other}` is not a barrier scope (run-start, since-barrier)"
        )),
    }
}

/// The Metal executor arms (`DESIGN.md` §9.2, §11.1–11.3; `--arena
/// --arena-layout --packed --replay --replay-barriers --replay-barrier-scope
/// --pass-plans`): installed on the device at load,
/// before the model. Instruments and A/B arms — none is the shipping path.
#[derive(Options, Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(default)]
pub struct ExecutorConfig {
    /// The activation arena (§9.2), recorded from the first decode steps (one skipped, two recorded); not under a speculative body.
    #[opt(bool, path = "executor.arena.enabled", status = "arm", facets = ["produce", "liveness"], flows = ["commands-out", "memory-return"])]
    pub arena: bool,
    /// The layout of the arena and of the pass-plan cache's plans: `packed` (values with disjoint liveness share a slot) or `non-aliasing` (the reference layout, every value its own slot).
    #[opt(flag = "arena-layout", parser = "parse_arena_layout", variants = ["packed", "non-aliasing"], path = "executor.arena.layout", status = "arm")]
    pub arena_layout: String,
    /// Every decode dispatch through its `_packed` entry point with the constants pool (implied by `replay`).
    #[opt(bool, path = "executor.packed", status = "arm", facets = ["produce"], flows = ["commands-out"])]
    pub packed: bool,
    /// The ICB replay executor (#146/#169): the stable packed subset of a step recorded into an indirect command buffer and replayed; implies `packed` and the arena.
    #[opt(bool, path = "executor.replay.enabled", status = "arm", facets = ["produce"], flows = ["commands-out"])]
    pub replay: bool,
    /// Under `replay`: whether candle's own barrier still fires at a replayed position — `always` or `skip-replayed` (§11.3r).
    #[opt(flag = "replay-barriers", parser = "parse_replay_barriers", variants = ["always", "skip-replayed"], path = "executor.replay.barriers", status = "arm", facets = ["signal"], flows = ["commands-out"])]
    pub replay_barriers: String,
    /// Under `replay`: how far back an ICB command's barrier scan looks — `run-start` or `since-barrier`.
    #[opt(flag = "replay-barrier-scope", parser = "parse_barrier_scope", variants = ["run-start", "since-barrier"], path = "executor.replay.barrier_scope", status = "arm", facets = ["signal"], flows = ["commands-out"])]
    pub replay_barrier_scope: String,
    /// The pass-keyed plan cache (§9.2aa): every forward pass served from one arena base by a plan cached per pass shape. On by default; `--pass-plans false` runs every pass on the buffer pool. `arena` or `replay` select the single-key arena instead.
    #[opt(flag = "pass-plans", bool, path = "executor.pass_plans", status = "advanced", facets = ["produce"], flows = ["commands-out"])]
    pub pass_plans: bool,
}

impl ExecutorConfig {
    /// Why the EOS gate cannot run under these arms, or `None`. The replay
    /// executor encodes recorded dispatches, which the gate's binding does
    /// not reach, and `Device::eos_gate_begin` refuses an installed executor.
    pub fn refuses_eos_gate(&self) -> Option<&'static str> {
        self.replay.then_some(
            "the ICB replay executor is installed, and the gate reaches directly encoded \
             dispatches only",
        )
    }

    /// Whether the pass-plan cache is installed: asked for, and not
    /// superseded by the single-key arena `arena`/`replay` record from the
    /// decode loop's legacy step bracket, which the cache's pass brackets
    /// replace (`MetalDevice::begin_pass` runs that bracket only with the
    /// cache off).
    pub fn pass_plans_armed(&self) -> bool {
        self.pass_plans && !self.arena && !self.replay
    }

    /// The arms for a report line, from the group that was installed.
    pub fn line(&self) -> String {
        format!(
            "arena={} packed={} replay={} pass_plans={}",
            if self.arena || self.replay {
                self.arena_layout.as_str()
            } else {
                "off"
            },
            self.packed || self.replay,
            if self.replay {
                format!("{}/{}", self.replay_barriers, self.replay_barrier_scope)
            } else {
                "off".to_string()
            },
            match (self.pass_plans_armed(), self.arena_layout.as_str()) {
                (true, "non-aliasing") => "true/non-aliasing",
                (true, _) => "true",
                (false, _) => "false",
            }
        )
    }
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        ExecutorConfig {
            arena: false,
            arena_layout: "packed".into(),
            packed: false,
            replay: false,
            replay_barriers: "skip-replayed".into(),
            replay_barrier_scope: "run-start".into(),
            pass_plans: true,
        }
    }
}

// ---- memory ---------------------------------------------------------------------

/// The memory budgets: the device fraction the process may claim
/// (`MemoryBudget` admission on LFM2, the streamed tier's slot budget on
/// Qwen4-exp), the KV ceiling and how the LFM2 slab grows toward it, the
/// pool's free-list budget, the Qwen tier's slots and reserve.
#[derive(Options, Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(default)]
pub struct MemoryConfig {
    /// The fraction of Metal's `recommendedMaxWorkingSetSize` this process may claim (§9.5d); 0 turns LFM2's admission off.
    #[opt(flag = "memory-fraction", path = "memory.fraction", status = "stable", cli = "memory-fraction", aliases = ["gpu-memory-utilization"])]
    pub memory_fraction: f64,
    /// KV positions per request (prompt + generation); a request past it is refused.
    #[opt(flag = "kv-capacity", path = "context.max_tokens", status = "stable", cli = "ctx-size", short = 'c', aliases = ["max-model-len"])]
    pub kv_capacity: usize,
    /// LFM2: positions the KV slab grows by when a pass would overrun it (a measurement knob; the stream is the same).
    #[opt(flag = "kv-bucket", path = "context.kv.grow_bucket", status = "arm")]
    pub kv_bucket: usize,
    /// LFM2: the fraction of the working set a KV grow may take the allocation to (§9.5c); none = the admission's.
    #[opt(
        flag = "kv-grow-fraction",
        path = "context.kv.grow_fraction",
        status = "advanced"
    )]
    pub kv_grow_fraction: Option<f64>,
    /// The buffer pool's free-list budget in MiB (256 ships; 1024 under a speculative body); none = the default.
    #[opt(
        flag = "free-budget-mb",
        path = "memory.pool_free_mb",
        status = "advanced"
    )]
    pub free_budget_mb: Option<usize>,
    /// Qwen4-exp `--slots auto`: KV tokens the slot budget reserves for (none = `kv_capacity`).
    #[opt(
        flag = "kv-reserve-tokens",
        path = "memory.kv_reserve_tokens",
        status = "advanced"
    )]
    pub kv_reserve_tokens: Option<usize>,
    /// Qwen4-exp `--experts stream`: streamed experts per layer per projection — a count, or `auto` (from the working set at `memory_fraction`).
    #[opt(flag = "slots", parser = "parse_slots", variants = ["auto", "<count>"], path = "weights.experts.slots", status = "stable", cli = "slots", facets = ["buffer"], flows = ["weights-in"])]
    pub slots: String,
    /// Admit every load against the machine before anything is allocated: its predicted peak against the memory the host has available now, less a reserve for the OS and the UI and what other lloom processes have reserved and not yet taken (memory.md §9.10). False loads unadmitted.
    #[opt(bool, path = "memory.machine.gate", status = "advanced")]
    pub machine_gate: bool,
    /// Load even when the machine gate refuses; the reservation is still recorded, so other lloom processes see it.
    #[opt(bool, path = "memory.machine.force", status = "stable", cli = "force")]
    pub force: bool,
    /// GB the machine gate keeps for macOS and the UI; none = max(4 GiB, the machine's memory less the Metal device's recommended working set), or max(4 GiB, 10 % of the machine's memory) with no Metal device.
    #[opt(path = "memory.machine.reserve_gb", status = "advanced")]
    pub machine_reserve_gb: Option<f64>,
    /// The directory of the machine-wide registry of lloom processes' reservations; none = ~/.cache/lloom/memory.
    #[opt(path = "memory.machine.registry", status = "advanced")]
    pub machine_registry: Option<std::path::PathBuf>,
    /// GB the machine gate takes as available instead of reading the host, for tests and dry runs; none = the host's reading.
    #[opt(path = "memory.machine.available_gb", status = "instrument")]
    pub machine_available_gb: Option<f64>,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        MemoryConfig {
            memory_fraction: 0.65,
            kv_capacity: 8704,
            kv_bucket: lloom::models::lfm2::DEFAULT_KV_BUCKET,
            kv_grow_fraction: None,
            free_budget_mb: None,
            kv_reserve_tokens: None,
            slots: "auto".into(),
            machine_gate: true,
            force: false,
            machine_reserve_gb: None,
            machine_registry: None,
            machine_available_gb: None,
        }
    }
}

impl MemoryConfig {
    /// The machine gate these options describe (`lloom_tensor::machine`).
    pub fn machine_gate(&self) -> lloom_tensor::machine::Gate {
        let bytes = |gb: f64| (gb.max(0.0) * 1e9) as u64;
        lloom_tensor::machine::Gate::new(lloom_tensor::machine::GateConfig {
            reserve: self.machine_reserve_gb.map(bytes),
            registry: self.machine_registry.clone(),
            available: self.machine_available_gb.map(bytes),
            total: None,
            working_set: None,
            force: self.force,
            off: !self.machine_gate,
        })
    }
}

// ---- what to load, and the front-ends' own keys -----------------------------------

/// `auto` or a count of slots.
fn parse_slots(v: &str) -> Result<String, String> {
    if v.trim().eq_ignore_ascii_case("auto") {
        return Ok("auto".into());
    }
    lloom_tensor::options::parse_count::<usize>(v)
        .map(|n| n.to_string())
        .map_err(|e| format!("{e}, or `auto`"))
}
fn parse_rows(v: &str) -> Result<String, String> {
    lloom_tensor::options::parse_arm(v, &["auto", "pool", "pool-per-row", "private"])
        .map(str::to_string)
}
fn parse_device(v: &str) -> Result<String, String> {
    match v.trim().to_ascii_lowercase().as_str() {
        "auto" => Ok("auto".into()),
        "cpu" => Ok("cpu".into()),
        other => Err(format!(
            "`{other}` is not a device (auto: the GPU this build has, else the CPU; cpu)"
        )),
    }
}
fn parse_experts(v: &str) -> Result<String, String> {
    match v.trim().to_ascii_lowercase().as_str() {
        s @ ("stream" | "resident") => Ok(s.into()),
        other => Err(format!("`{other}` is not `stream` or `resident`")),
    }
}
fn parse_loader(v: &str) -> Result<String, String> {
    match v.trim().to_ascii_lowercase().as_str() {
        s @ ("pread" | "io") => Ok(s.into()),
        other => Err(format!("`{other}` is not `pread` or `io`")),
    }
}
fn parse_moe_act(v: &str) -> Result<String, String> {
    match v.trim().to_ascii_lowercase().as_str() {
        s @ ("auto" | "f32" | "f16") => Ok(s.into()),
        other => Err(format!("`{other}` is not `auto`, `f32` or `f16`")),
    }
}
fn parse_prefill_mode(v: &str) -> Result<String, String> {
    match v.trim().to_ascii_lowercase().as_str() {
        s @ ("chunk" | "replay") => Ok(s.into()),
        other => Err(format!("`{other}` is not `chunk` or `replay`")),
    }
}

/// What to load and on what: the checkpoint, its served name and
/// tokenizer, the device, and the family's load-time options (Qwen4-exp's
/// streamed expert tier and PLE; LFM2's FlashDecoding chunk).
#[derive(Options, Serialize, Deserialize, Debug, Clone, PartialEq)]
#[options(v0 = false)]
#[serde(default)]
pub struct LoadConfig {
    /// The model: an LFM2 checkpoint directory (config.json, tokenizer.json, *.safetensors, chat_template.jinja) or a Qwen4-exp GGUF (the first shard, `-00001-of-0000N`, its siblings beside it, or a directory holding them); the family is read from the path. Default: the LFM2.5-2.6B snapshot in the HF cache.
    #[opt(
        flag = "model",
        alias = "model-dir",
        env = "LFM2_MODEL_DIR",
        path = "model.path",
        status = "stable",
        cli = "model"
    )]
    pub model: Option<std::path::PathBuf>,
    /// The `model` id `/v1/models` reports and requests may name. Default: the checkpoint's Hub id when it is an HF-cache snapshot.
    #[opt(
        flag = "model-name",
        path = "model.served_name",
        status = "stable",
        cli = "served-model-name"
    )]
    pub model_name: Option<String>,
    /// The GGUF families (Qwen4-exp, Qwen3.5): tokenizer.json, with chat_template.jinja beside it. Default: Qwen4-exp's is the HF cache's Qwen/Qwen3.8-Flash-Next snapshot; Qwen3.5's the base model's snapshot the quantizer recorded, else the GGUF's own vocabulary.
    #[opt(
        flag = "tokenizer",
        path = "model.tokenizer",
        status = "advanced",
        cli = "tokenizer"
    )]
    pub tokenizer: Option<std::path::PathBuf>,
    /// Explicit GGUF media companion (d1-omni vision/audio). No sidecar variant is guessed. Safetensors checkpoints keep media in their checkpoint and do not consume this option.
    #[opt(
        flag = "mmproj",
        path = "model.mmproj",
        status = "advanced",
        cli = "mmproj"
    )]
    pub mmproj: Option<std::path::PathBuf>,
    /// The template `/v1/systemone` answers with. Unset (or `auto`): a native decision checkpoint's own binding (d1-3B, d1-omni), else decisions are refused by name. `generic`: one label token per option through the model's own chat template, thinking off, uncalibrated (every answer reports its label mass). `lfm2-d1` | `lfm2-d1-omni`: the engine-shipped d1 template. A path: that systemone Jinja template.
    #[opt(
        flag = "decision-template",
        path = "decision.template",
        status = "advanced",
        cli = "decision-template"
    )]
    pub decision_template: Option<String>,
    /// Load no vision or audio tower (a GGUF's mmproj, a VL checkpoint's tower): for text workloads, at the tower's memory. Image requests are then refused by name, and `/v1/models` lists text input only.
    #[opt(
        bool,
        flag = "text-only",
        path = "model.text_only",
        status = "advanced",
        cli = "text-only"
    )]
    pub text_only: bool,
    /// Compile the prefill pipelines at load (off: the first prompt compiles them inside its own time to first token).
    #[opt(bool, flag = "warmup", path = "model.warmup", status = "advanced", switches = ["no-warmup=false"])]
    pub warmup: bool,
    /// The device: `auto` (the GPU this build has, else the CPU) or `cpu`.
    #[opt(flag = "device", parser = "parse_device", variants = ["auto", "cpu"], path = "backend.device", status = "stable", cli = "device", switches = ["cpu=cpu"])]
    pub device: String,
    /// Qwen4-exp: how the routed experts are held — `stream` (a slot pool per layer filled from the checkpoint on demand) or `resident` (every expert in memory). Qwen3.5's MoE loads resident whatever it says, with a load warning.
    #[opt(flag = "experts", parser = "parse_experts", variants = ["stream", "resident"], path = "weights.experts.placement", status = "stable", cli = "experts", facets = ["source"], flows = ["weights-in"])]
    pub experts: String,
    /// Qwen4-exp, streamed experts: the loader primitive, `pread` (a read and a compute-encoder copy) or `io` (`MTLIOCommandQueue`).
    #[opt(flag = "loader", parser = "parse_loader", variants = ["pread", "io"], path = "weights.experts.loader", status = "advanced", facets = ["transport"], flows = ["weights-in"])]
    pub loader: String,
    /// Qwen4-exp: rows of the N-gram PLE hot-row cache (0 = read the memory map every gather).
    #[opt(flag = "ple-cache-rows", path = "models.qwen4exp.ple.cache_rows", status = "advanced", facets = ["buffer"], flows = ["host-feed-in"])]
    pub ple_cache_rows: usize,
    /// Qwen4-exp: load the N-gram PLE (off: a measurement arm; the digests belong to the PLE-on model).
    #[opt(bool, flag = "ple", path = "models.qwen4exp.ple.enabled", status = "reference", switches = ["no-ple=false"], facets = ["source"], flows = ["host-feed-in"])]
    pub ple: bool,
    /// Qwen4-exp: the routed experts' activation dtype — `auto` (`f16` under the `fast` contract, `f32` under `deterministic`), `f32` (the byte-identical routed arm) or `f16`.
    #[opt(flag = "moe-act", parser = "parse_moe_act", variants = ["auto", "f32", "f16"], path = "models.qwen4exp.moe.activation_dtype", status = "advanced")]
    pub moe_act: String,
    /// Qwen4-exp, streamed experts: the expert hotlist file — read at load to warm every layer's pool before the first request (a missing file warms nothing), written at shutdown with the usage since load merged in.
    #[opt(flag = "hotlist", path = "weights.experts.hotlist.path", status = "advanced", facets = ["buffer"], flows = ["weights-in"])]
    pub hotlist: Option<std::path::PathBuf>,
    /// Qwen4-exp, streamed experts: the expert prefetcher (the tier's reads off the encoding thread, ahead of the layer).
    #[opt(bool, flag = "prefetch", path = "weights.experts.prefetch.enabled", status = "advanced", facets = ["lead"], flows = ["weights-in"])]
    pub prefetch: bool,
    /// Qwen4-exp, streamed experts on the `io` loader: the in-flight residency hints into free slots (measured not to pay on a pool smaller than the tile).
    #[opt(bool, flag = "io-hints", path = "weights.experts.io_hints", status = "arm", facets = ["lead"], flows = ["weights-in"])]
    pub io_hints: bool,
    /// Qwen4-exp: how a prompt is prefilled — `chunk` (the chunked DeltaNet and QSA passes) or `replay` (the decode recurrence replayed per token; a reference arm).
    #[opt(flag = "prefill-mode", parser = "parse_prefill_mode", variants = ["chunk", "replay"], path = "models.qwen4exp.prefill_mode", status = "reference")]
    pub prefill_mode: String,
    /// Qwen4-exp, streamed experts with a hotlist: pin this many experts per layer from the hotlist (never evicted); 0 = none.
    #[opt(flag = "pin", path = "weights.experts.hotlist.pin", status = "advanced", facets = ["buffer"], flows = ["weights-in"])]
    pub pin: usize,
    /// LFM2 under the FlashDecoding attention: pages per FlashDecoding chunk (1, the shipped value, is a 256-key chunk per partial threadgroup); a probe of the partial kernel's geometry at long context.
    #[opt(
        flag = "flash-k",
        path = "kernels.attention.flash.pages_per_chunk",
        status = "arm"
    )]
    pub flash_k: Option<usize>,
}

impl Default for LoadConfig {
    fn default() -> Self {
        LoadConfig {
            model: None,
            model_name: None,
            tokenizer: None,
            mmproj: None,
            decision_template: None,
            text_only: false,
            warmup: true,
            device: "auto".into(),
            experts: "stream".into(),
            loader: "pread".into(),
            ple_cache_rows: 0,
            ple: true,
            moe_act: "auto".into(),
            hotlist: None,
            prefetch: true,
            io_hints: false,
            prefill_mode: "chunk".into(),
            pin: 0,
            flash_k: None,
        }
    }
}

impl LoadConfig {
    /// The CPU device was asked for.
    pub fn cpu(&self) -> bool {
        self.device == "cpu"
    }
}

/// Concurrent sequences: how many decode together and how their state is
/// held (`lloom serve`, `lloom bench`).
#[derive(Options, Serialize, Deserialize, Debug, Clone, PartialEq)]
#[options(v0 = false)]
#[serde(default)]
pub struct SchedulerConfig {
    /// The batch width: how many requests decode together, one weight sweep per step for all of them. `1` is the single-sequence loop; above it the continuous-batching scheduler.
    #[opt(flag = "batch", path = "scheduler.max_sequences", status = "stable", cli = "batch", aliases = ["max-num-seqs"])]
    pub max_sequences: usize,
    /// How the rows of a batch hold their state: `auto` (`pool` at a batch), `pool` (one shared paged pool, the batch in one dispatch per kernel), `pool-per-row` (the same pool, the per-row loop) or `private` (a cache per row).
    #[opt(flag = "rows", parser = "parse_rows", variants = ["auto", "pool", "pool-per-row", "private"], path = "scheduler.rows", status = "advanced")]
    pub rows: String,
    /// Read `/v1/systemone` requests that wait together for the model as one tree, the reference's packing of single-question requests (the `fast` contract only: under `deterministic` each request is read alone).
    #[opt(bool, flag = "decision-pack", path = "scheduler.decision_pack", status = "advanced", switches = ["no-decision-pack=false"])]
    pub decision_pack: bool,
    /// Read a `/v1/systemone` request's questions as one ragged batch over its state, the state prefilled once and kept for the next request over it (the `fast` contract only: under `deterministic` the questions are read one after another from a restore of the state).
    #[opt(bool, flag = "decision-ragged", path = "scheduler.decision_ragged", status = "advanced", switches = ["no-decision-ragged=false"])]
    pub decision_ragged: bool,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        SchedulerConfig {
            max_sequences: 1,
            rows: "auto".into(),
            decision_pack: true,
            decision_ragged: true,
        }
    }
}

/// Prefix reuse across turns and requests.
#[derive(Options, Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
#[options(v0 = false)]
#[serde(default)]
pub struct PrefixConfig {
    /// Reuse the cached prefix across turns or requests; unset = the front-end's own.
    #[opt(bool, flag = "reuse-prefix", path = "context.prefix.reuse", status = "stable", cli = "prefix-reuse", switches = ["no-reuse=false"])]
    pub reuse: Option<bool>,
    /// Share a request's prompt prefix from another row's cache, by page (the pool rows only).
    #[opt(
        bool,
        flag = "share-prefix",
        path = "context.prefix.share",
        status = "advanced"
    )]
    pub share: bool,
}

/// The thinking block a chat template opens.
#[derive(Options, Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
#[options(v0 = false)]
#[serde(default)]
pub struct ReasoningConfig {
    /// Let the model think (the template's `enable_thinking`, or the block its generation prompt opens); unset = the template's own rendering.
    #[opt(bool, flag = "think", path = "reasoning.enabled", status = "stable", cli = "think", switches = ["no-think=false"])]
    pub enabled: Option<bool>,
    /// How long the model thinks: the chat template's `reasoning_effort`, in the template's own values.
    #[opt(
        flag = "reasoning-effort",
        path = "reasoning.effort",
        status = "stable",
        cli = "reasoning-effort"
    )]
    pub effort: Option<String>,
}

fn parse_missing_reasoning(v: &str) -> Result<String, String> {
    lloom_tensor::options::parse_arm(v, &["as-sent", "log"]).map(str::to_string)
}

fn parse_incomplete_reasoning(v: &str) -> Result<String, String> {
    lloom_tensor::options::parse_arm(v, &["text", "reasoning", "recover"]).map(str::to_string)
}

fn parse_keep_logprobs(v: &str) -> Result<String, String> {
    lloom_tensor::options::parse_arm(v, &["none", "sampled", "full"]).map(str::to_string)
}

/// The conversation's history as the front end keeps it: the sampled ids of
/// each reply, and what is kept beside them.
#[derive(Options, Serialize, Deserialize, Debug, Clone, PartialEq)]
#[options(v0 = false)]
#[serde(default)]
pub struct SessionConfig {
    /// Keep past replies' reasoning in the history the next turn renders (the chat template's `preserve_thinking`): unset keeps it, as `true` does, wherever the template has the switch (its own rendering where it has none); `false` strips it (opt-in).
    #[opt(bool, flag = "persist-thinking", path = "session.persist_thinking", status = "stable", cli = "persist-thinking", switches = ["strip-thinking=false"])]
    pub persist_thinking: Option<bool>,
    /// A past reply of ours sent back without the reasoning the model wrote: `as-sent` renders the message as the client sent it; `log` restores the reasoning from the session log.
    #[opt(flag = "missing-reasoning", parser = "parse_missing_reasoning", variants = ["as-sent", "log"], path = "session.missing_reasoning", status = "advanced")]
    pub missing_reasoning: String,
    /// What the session log keeps beside each reply's sampled ids: `none`; `sampled`, the sampled token's log-probability; `full`, every position's log-probabilities over the vocabulary (for a trainer that embeds the engine).
    #[opt(flag = "keep-logprobs", parser = "parse_keep_logprobs", variants = ["none", "sampled", "full"], path = "session.keep_logprobs", status = "advanced")]
    pub keep_logprobs: String,
    /// A past reply of ours whose run ended inside its reasoning block (`max_tokens`, a stop string, a cancellation): `reasoning` (the default) gives it to the template as the open reasoning it is, which the template closes or drops; `text` renders it as the text the client got; `recover` renders it as text and places the turn checkpoint before the divergence that makes.
    #[opt(flag = "incomplete-reasoning", parser = "parse_incomplete_reasoning", variants = ["text", "reasoning", "recover"], path = "session.incomplete_reasoning", status = "advanced")]
    pub incomplete_reasoning: String,
}

impl Default for SessionConfig {
    fn default() -> Self {
        SessionConfig {
            persist_thinking: None,
            missing_reasoning: "as-sent".into(),
            keep_logprobs: "none".into(),
            incomplete_reasoning: "reasoning".into(),
        }
    }
}

/// How a prompt is encoded.
#[derive(Options, Serialize, Deserialize, Debug, Clone, PartialEq)]
#[options(v0 = false)]
#[serde(default)]
pub struct PromptConfig {
    /// Encode the prompt with the checkpoint's BOS.
    #[opt(bool, flag = "bos", path = "prompt.add_bos", status = "advanced", switches = ["no-bos=false"])]
    pub add_bos: bool,
}

impl Default for PromptConfig {
    fn default() -> Self {
        PromptConfig { add_bos: true }
    }
}

/// What a front-end reports beside its output.
#[derive(Options, Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
#[options(v0 = false)]
#[serde(default)]
pub struct ReportConfig {
    /// Print each turn's diagnostics: the first token's split, the cadence, the hot thread, the churn, the pacer's depth, the KV slab, the speculative rounds.
    #[opt(
        bool,
        flag = "verbose",
        short = 'v',
        path = "diagnostics.verbose",
        status = "stable",
        cli = "verbose",
        needs = "profile",
        needs_part = "the first token's split and the step-time percentiles"
    )]
    pub verbose: bool,
    /// Sample the memory gauges every 10 ms around each turn or cell and report their peaks (a gauge run: it perturbs the timing beside it).
    #[opt(
        bool,
        flag = "mem-sample",
        path = "diagnostics.memory_sample",
        status = "instrument",
        cli = "mem-sample"
    )]
    pub memory_sample: bool,
}

/// The pipette sidecar's contract.
#[derive(Options, Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
#[options(v0 = false)]
#[serde(default)]
pub struct PipetteConfig {
    /// A file whose text is tiled into the throughput cells' prompts; unset = the pipette's env seed, then a built-in passage.
    #[opt(
        flag = "prompt-seed-file",
        path = "server.pipette.prompt_seed_file",
        status = "advanced"
    )]
    pub prompt_seed_file: Option<std::path::PathBuf>,
}

/// `lloom serve`'s socket.
#[derive(Options, Serialize, Deserialize, Debug, Clone, PartialEq)]
#[options(v0 = false)]
#[serde(default)]
pub struct ServerConfig {
    /// The bind address (loopback: a sidecar, not a public server).
    #[opt(flag = "host", path = "server.host", status = "stable", cli = "host")]
    pub host: String,
    /// The port; 0 picks a free one (the bound port is in the ready marker).
    #[opt(flag = "port", path = "server.port", status = "stable", cli = "port")]
    pub port: u16,
    #[opt(nested)]
    pub pipette: PipetteConfig,
}

impl Default for ServerConfig {
    fn default() -> Self {
        ServerConfig {
            host: "127.0.0.1".into(),
            port: 0,
            pipette: PipetteConfig::default(),
        }
    }
}

fn parse_bench_mode(v: &str) -> Result<String, String> {
    match v.trim().to_ascii_lowercase().as_str() {
        s @ ("cells" | "reuse" | "concurrent") => Ok(s.into()),
        other => Err(format!("`{other}` is not `cells`, `reuse` or `concurrent`")),
    }
}

/// `lloom bench`: the cells, and the multi-sequence modes.
#[derive(Options, Serialize, Deserialize, Debug, Clone, PartialEq)]
#[options(v0 = false)]
#[serde(default)]
pub struct BenchConfig {
    /// `cells`, or a multi-sequence mode on LFM2: `reuse` (a two-turn conversation through prefix reuse against a fresh prefill) or `concurrent` (several conversations on one model, each stream checked against its solo run).
    #[opt(flag = "mode", parser = "parse_bench_mode", variants = ["cells", "reuse", "concurrent"], path = "bench.mode", status = "stable", cli = "mode")]
    pub mode: String,
    /// The cells, comma-separated: `prefill:P`, `decode:P:D`, `e2e:P:D`, `memory:P:D` (P prompt tokens, D decode tokens).
    #[opt(flag = "cells", path = "bench.cells", status = "stable", cli = "cells")]
    pub cells: String,
    /// Runs per cell (the first is the cold one).
    #[opt(flag = "runs", path = "bench.runs", status = "stable", cli = "runs")]
    pub runs: usize,
    /// `concurrent`: how many sequences (the prompts cycle a varied set of eight).
    #[opt(
        flag = "sequences",
        path = "bench.concurrent.sequences",
        status = "advanced"
    )]
    pub sequences: usize,
    /// `concurrent`: drive the sequences through the batched-forward driver, against each prompt run solo.
    #[opt(
        bool,
        flag = "batched",
        path = "bench.concurrent.batched",
        status = "advanced"
    )]
    pub batched: bool,
    /// `concurrent` batched: the neighbour-independence arm (the rows re-run as two separately batched halves).
    #[opt(bool, flag = "independence-check", path = "bench.concurrent.independence_check", status = "advanced", switches = ["no-independence-check=false"])]
    pub independence_check: bool,
    /// `concurrent` batched: skip the solo arm (the identity columns then read `false`).
    #[opt(
        bool,
        flag = "skip-solo",
        path = "bench.concurrent.skip_solo",
        status = "advanced"
    )]
    pub skip_solo: bool,
    /// `concurrent` batched: exit non-zero when the rows gate reads `fail` (off: report it and exit 0).
    #[opt(
        bool,
        flag = "fail-on-gate",
        env = "LLOOM_FAIL_ON_GATE",
        path = "bench.fail_on_gate",
        status = "advanced"
    )]
    pub fail_on_gate: bool,
    /// `concurrent`: tokens per visit on the time-sliced interleave.
    #[opt(flag = "chunk", path = "bench.concurrent.chunk", status = "advanced")]
    pub chunk: usize,
    /// `concurrent`: sequence `i` arrives at round (interleave) or step (batched) `i × N`; 0 admits every sequence at once.
    #[opt(
        flag = "stagger",
        path = "bench.concurrent.stagger",
        status = "advanced"
    )]
    pub stagger: usize,
    /// `concurrent`: give the last sequence a synthetic prompt of exactly this many body tokens; 0 keeps the corpus prompt.
    #[opt(
        flag = "long-prompt-tokens",
        path = "bench.concurrent.long_prompt_tokens",
        status = "advanced"
    )]
    pub long_prompt_tokens: usize,
    /// `concurrent`: which arm runs first — `solo`, or `pipelined` (a drafter) / `batched` — so a rotation can alternate the order.
    #[opt(flag = "first", path = "bench.concurrent.first", status = "advanced")]
    pub first: String,
    /// `concurrent` under a drafter: write every round of both arms as CSV.
    #[opt(
        flag = "timeline",
        path = "bench.concurrent.timeline",
        status = "instrument"
    )]
    pub timeline: Option<std::path::PathBuf>,
}

impl Default for BenchConfig {
    fn default() -> Self {
        BenchConfig {
            mode: "cells".into(),
            cells: "prefill:512,decode:512:100,e2e:8:64,memory:512:100".into(),
            runs: 2,
            sequences: 2,
            batched: false,
            independence_check: true,
            skip_solo: false,
            fail_on_gate: false,
            chunk: 16,
            stagger: 0,
            long_prompt_tokens: 0,
            first: "solo".into(),
            timeline: None,
        }
    }
}

/// Every option group the engine reads, as one tree: the dotted keys are
/// `<group>.<key>` (`decode.readback.eos`, `tuning.qmv.q6k_nr0`), the JSON
/// is one object per group, and `lloom config` prints the resolved tree back
/// in this shape.
#[derive(Options, Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
#[serde(default)]
pub struct EngineOptions {
    /// What to load and on what (first: its flags lead `--help`).
    #[opt(nested)]
    pub load: LoadConfig,
    /// The decode loop: the readback arm, the pacer, the prefill.
    #[opt(nested)]
    pub decode: DecodeConfig,
    /// Every tuning knob of the kernels, the device, the pool, the model ports.
    #[opt(nested)]
    pub tuning: TuningOptions,
    /// The model's axes: a cache preset and per-axis overrides.
    #[opt(nested)]
    pub model: ModelConfig,
    /// The sampling defaults.
    #[opt(nested)]
    pub sampling: SamplingConfig,
    /// The speculative body.
    #[opt(nested)]
    pub speculative: SpeculativeConfig,
    /// The Metal executor arms.
    #[opt(nested)]
    pub executor: ExecutorConfig,
    /// The memory budgets.
    #[opt(nested)]
    pub memory: MemoryConfig,
    /// Concurrent sequences (`serve`, `bench` flatten its flags).
    #[opt(nested, no_args)]
    pub scheduler: SchedulerConfig,
    /// Prefix reuse.
    #[opt(nested, no_args)]
    pub prefix: PrefixConfig,
    /// The thinking block (`chat`).
    #[opt(nested, no_args)]
    pub reasoning: ReasoningConfig,
    /// Prompt encoding (`chat`).
    #[opt(nested, no_args)]
    pub prompt: PromptConfig,
    /// The conversation's history (`chat`, `serve`).
    #[opt(nested, no_args)]
    pub session: SessionConfig,
    /// What a front-end reports.
    #[opt(nested, no_args)]
    pub report: ReportConfig,
    /// `serve`'s socket and the sidecar contract.
    #[opt(nested, no_args)]
    pub server: ServerConfig,
    /// `bench`'s cells and modes.
    #[opt(nested, no_args)]
    pub bench: BenchConfig,
}

impl EngineOptions {
    /// The refusals of every group, once: each group's own, then the
    /// static couplings by id ([`crate::couplings::refuse`]).
    pub fn check(&self) -> Result<(), String> {
        self.decode.check()?;
        self.sampling.check()?;
        crate::couplings::refuse(self)
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

    /// The default is the flow: the crown's ring with the adaptive commit
    /// and the EOS gate, and the arm it resolves to names no clamp.
    #[test]
    fn the_default_is_the_crown() {
        let cfg = DecodeConfig::default();
        assert_eq!(cfg.readback.arm(), ReadbackArm::Auto);
        assert_eq!(cfg.readback.ring_arm(), RingArm::CROWN);
        assert!(cfg.readback.eos_gate, "the EOS gate is on by default");
        assert_eq!(cfg.readback.adaptive_clamp(), None);
        assert!(cfg.check().is_ok());
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
        cfg.set("prefill.spark", "progressive:1,64/tile:64")
            .unwrap();
        cfg.set("in_flight", "3").unwrap();
        assert_eq!(cfg.readback.eos, EosKind::HostDrain);
        assert_eq!(cfg.readback.ring_arm().wake, 2);
        assert_eq!(cfg.prefill.schedule(), Some((vec![1, 16, 32, 64], 64)));
        assert!(cfg.set("prefill.spark", "8").is_err());
        cfg.merge_json(
            &serde_json::json!({"readback": {"eos": "gpu-batched", "commit_budget": "steps:4"}}),
        )
        .unwrap();
        assert_eq!(cfg.readback.eos, EosKind::GpuBatched);
        assert_eq!(cfg.readback.commit_budget.period(), 4);
        cfg.merge_json(&serde_json::json!({"readback": "blocking"}))
            .unwrap();
        assert_eq!(cfg.readback.kind, ReadbackKind::Blocking);
        let dumped = serde_json::to_value(&cfg).unwrap();
        let mut back = DecodeConfig::default();
        back.merge_json(&dumped).unwrap();
        assert_eq!(back, cfg);
        assert!(
            cfg.check().is_ok(),
            "the cadence knobs are ignored on the blocking arm, not refused"
        );
        cfg.set("readback.eos_gate", "true").unwrap();
        cfg.set("readback.adaptive_commit", "false").unwrap();
        assert!(
            cfg.check().is_ok(),
            "the EOS gate rides the adaptive commit, not refused without it"
        );
        cfg.set("readback.eos_gate", "false").unwrap();
        assert!(!cfg.readback.eos_gate, "the ungated arm stays selectable");
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

    /// `stamps_each_token` answers what the ring itself answers, on every
    /// EOS read at both cadences (the CPU ring: the rule is the strategy's,
    /// not the device's).
    #[test]
    fn stamps_each_token_is_the_rings_own_answer() {
        use lloom::generation::{ReadbackStrategy, Ring};
        let reads = [
            EosRead::Blocking,
            EosRead::Poll(PollMechanism::default()),
            EosRead::HostOnDrain,
            EosRead::GpuBatched,
            EosRead::GpuBatchedNonblocking,
        ];
        let mut per_token = 0;
        for eos in reads {
            let ring = Ring::new(&[2], &lloom_tensor::Device::Cpu, 64, eos).unwrap();
            for adaptive in [false, true] {
                let arm = RingArm {
                    adaptive,
                    eos,
                    ..RingArm::CROWN
                };
                assert_eq!(
                    stamps_each_token(arm),
                    ring.stamps_each_token(adaptive),
                    "{}",
                    arm.name()
                );
                per_token += stamps_each_token(arm) as usize;
            }
        }
        assert_eq!(
            per_token, 7,
            "poll and flag twice, the three deferred reads once"
        );
    }

    /// A ring arm that cannot carry the adaptive commit — the flag EOS read,
    /// or no lookahead — runs the fixed cadence and says why at load, naming
    /// the EOS gate that rides it; neither default-on knob is refused. The
    /// crown and an explicit off say nothing.
    #[test]
    fn an_arm_that_cannot_carry_the_adaptive_commit_runs_the_fixed_cadence() {
        let mut ring = ReadbackConfig::default();
        ReadbackConfig::apply_spelling(&mut ring, "ring").unwrap();
        assert!(
            ring.adaptive_commit,
            "the spelling leaves the knob as it was"
        );
        let flag_fixed = RingArm {
            eos: EosRead::Blocking,
            ..RingArm::FIXED
        };
        assert_eq!(ring.ring_arm(), flag_fixed);
        assert_eq!(ring.arm(), ReadbackArm::Ring(flag_fixed));
        let note = ring.adaptive_clamp().expect("the clamp is named");
        assert!(
            note.contains("ring lookahead=1 commit_steps=8")
                && note.contains("flag")
                && note.contains("without the EOS gate"),
            "{note}"
        );
        assert!(
            ring.eos_gate && ring.check().is_ok(),
            "the default gate is clamped, not refused"
        );
        ring.eos_gate = false;
        let note = ring.adaptive_clamp().expect("the clamp is named");
        assert!(!note.contains("EOS gate"), "{note}");

        let mut la0 = ReadbackConfig {
            lookahead: 0,
            ..ReadbackConfig::default()
        };
        assert_eq!(
            la0.ring_arm(),
            RingArm {
                lookahead: 0,
                ..RingArm::FIXED
            }
        );
        assert!(la0.adaptive_clamp().unwrap().contains("lookahead 1"));
        la0.adaptive_commit = false;
        assert_eq!(la0.adaptive_clamp(), None);

        // A deferred read carries the adaptive commit with or without the
        // fused publish: the loop stamps each token after an unfused step.
        let mut drain = ReadbackConfig::default();
        ReadbackConfig::apply_spelling(&mut drain, "ring-hostdrain").unwrap();
        assert!(drain.ring_arm().adaptive);
        drain.fused_publish = false;
        assert!(drain.ring_arm().adaptive);
        assert_eq!(drain.adaptive_clamp(), None);

        assert_eq!(ReadbackConfig::default().adaptive_clamp(), None);
        assert_eq!(ReadbackConfig::default().ring_arm(), RingArm::CROWN);
        let mut blocking = ReadbackConfig::default();
        ReadbackConfig::apply_spelling(&mut blocking, "blocking").unwrap();
        blocking.lookahead = 0;
        blocking.eos_gate = true;
        assert_eq!(blocking.adaptive_clamp(), None);
        assert!(
            blocking.check().is_ok(),
            "the blocking arm ignores the cadence knobs"
        );

        // The stream reader is taken on every spelling, cadence and publish:
        // every EOS read stamps on a grain its reader streams.
        for spelling in ReadbackConfig::SPELLINGS {
            for adaptive_commit in [true, false] {
                for fused_publish in [true, false] {
                    let mut reader = ReadbackConfig {
                        stream_reader: true,
                        adaptive_commit,
                        fused_publish,
                        ..ReadbackConfig::default()
                    };
                    ReadbackConfig::apply_spelling(&mut reader, spelling).unwrap();
                    assert_eq!(
                        reader.check(),
                        Ok(()),
                        "--readback {spelling} --stream-reader adaptive={adaptive_commit} \
                         fused={fused_publish}"
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod spark_tests {
    use super::parse_spark;

    #[test]
    fn a_fixed_bucket_takes_the_seq1_tail() {
        assert_eq!(parse_spark("16").unwrap(), (vec![16], 0));
        assert_eq!(parse_spark("1").unwrap(), (vec![1], 0));
        assert_eq!(parse_spark("64").unwrap(), (vec![64], 0));
    }

    #[test]
    fn a_width_off_the_buckets_is_refused() {
        assert!(parse_spark("4").is_err());
        assert!(parse_spark("8").is_err());
        assert!(parse_spark("100").is_err());
    }

    #[test]
    fn progressive_ramps_through_the_buckets() {
        assert_eq!(
            parse_spark("progressive:1,64").unwrap(),
            (vec![1, 16, 32, 64], 0)
        );
        assert_eq!(parse_spark("progressive:1,16").unwrap(), (vec![1, 16], 0));
        assert_eq!(
            parse_spark("progressive:16,64").unwrap(),
            (vec![16, 32, 64], 0)
        );
        assert_eq!(
            parse_spark("progressive:1,32").unwrap(),
            (vec![1, 16, 32], 0)
        );
        assert!(parse_spark("progressive:64,1").is_err());
        assert!(parse_spark("progressive:4,64").is_err());
        assert!(parse_spark("progressive:1").is_err());
    }

    #[test]
    fn the_bulk_tile_rides_either_ramp() {
        assert_eq!(parse_spark("16/tile:32").unwrap(), (vec![16], 32));
        assert_eq!(parse_spark("1/tile:64").unwrap(), (vec![1], 64));
        assert_eq!(
            parse_spark("progressive:1,64/tile:64").unwrap(),
            (vec![1, 16, 32, 64], 64)
        );
        assert!(parse_spark("16/tile:4").is_err());
        assert!(
            parse_spark("16/tile:1").is_err(),
            "1 is the tail, not a bulk tile"
        );
        assert!(parse_spark("16/tile:x").is_err());
    }
}
