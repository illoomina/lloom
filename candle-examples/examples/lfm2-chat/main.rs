//! LFM2 chat, the smallest useful shape.
//!
//! Reads a prompt from **stdin**, prints the reply to **stdout**, and prints a
//! rough decode throughput (tok/s) to **stderr**. Single turn, greedy by
//! default. No flags you have to learn — point it at a checkpoint and pipe a
//! prompt in.
//!
//!   echo "Explain how a hash map works." \
//!     | cargo run --release --features metal --example lfm2-chat
//!
//! The model directory defaults to the LFM2.5-2.6B snapshot in the HF cache;
//! override with `--model-dir <path>` or `LFM2_MODEL_DIR=<path>`.
//!
//! This is the "pytorchy" path: load config + weights, build the model and a KV
//! cache, then one `Session::decode` call. The loop itself lives in
//! `lloom::nn::decode` (`design/docs/decode-driver-design.md`, stage 1); what
//! this file selects is the readback strategy — `--readback blocking` (the
//! default, the host reads every token back) or `ring` / `ring-nonblocking`
//! (the token stays on the device) — and, on the ring arms, `--lookahead 1`,
//! which lets the host encode the next step before it has learned the previous
//! one's EOS. The perf harness is `lfm2-release`; this one is for reading.

use anyhow::{Context, Result};
use candle::{DType, Device};
use clap::Parser;
use lloom::generation::poll::PollMechanism;
use lloom::generation::{
    Blocking, CommitBudget, EosRead, LogitsProcessor, ReadbackStrategy, Ring, Sampling,
};
use lloom::models::lfm2::{Cache, ConvState, KvAppend, Lfm2Config, Model};
use lloom::models::lfm2_dspark::{DSparkConfig, DSparkDrafter, DSparkSpeculator};
use lloom::nn::{DecodeOptions, DecodeOut, IdentityProposer, Session, VarBuilder};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use tokenizers::Tokenizer;

mod concurrent;

/// A pre-allocated, lock-free single-producer/single-consumer ring of token ids —
/// the seam that moves the terminal write OFF the decode thread.
///
/// The decode loop (producer) drains its device ring on the hot thread as before,
/// then `push`es the ids here — atomic stores only, NO allocation, NO syscall, NO
/// notify (the notify was what spiked `step_waits` when this was tried as an mpsc
/// channel: a per-commit futex wake on the hot thread). A separate consumer thread
/// `pop`s at its own cadence, detokenizes, and writes stdout — so a slow terminal
/// stalls the CONSUMER, never the decode loop.
///
/// Fixed power-of-two capacity, sized once to cover the whole run, so `push` never
/// finds it full in practice (a full ring means the consumer is wedged on a dead
/// terminal — then, and only then, `push` spins, which is the correct place for a
/// sustained-backpressure stall to land). `head`/`tail` are the classic SPSC
/// cursors: the producer owns `tail`, the consumer owns `head`, and the
/// release/acquire pair is the happens-before that publishes each slot's write
/// ahead of the `tail` bump (§9.3's discipline, host-thread to host-thread this
/// time rather than GPU to host).
struct TokenSpsc {
    slots: Box<[std::cell::UnsafeCell<u32>]>,
    mask: usize,
    /// Next slot the producer writes; `Release`-published so the consumer that
    /// observes it sees the slot's write behind it.
    tail: AtomicUsize,
    /// Next slot the consumer reads.
    head: AtomicUsize,
    /// Producer sets this when the run is over; the consumer drains what remains
    /// and then exits.
    closed: AtomicBool,
}

// SAFETY: single-producer/single-consumer by contract — the producer only ever
// touches `tail` (+ the slot at `tail`), the consumer only `head` (+ the slot at
// `head`), and a slot is written by the producer strictly before its `tail`
// release and read by the consumer strictly after its `tail` acquire, so the two
// never touch the same slot concurrently. That makes the `UnsafeCell<u32>` sharing
// sound across the two threads.
unsafe impl Sync for TokenSpsc {}
unsafe impl Send for TokenSpsc {}

impl TokenSpsc {
    /// Capacity is rounded up to a power of two `>= cap.max(1024)` so the mask
    /// trick applies and a short run still gets a generous buffer.
    fn new(cap: usize) -> Self {
        let n = cap.max(1024).next_power_of_two();
        let slots = (0..n)
            .map(|_| std::cell::UnsafeCell::new(0u32))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self {
            slots,
            mask: n - 1,
            tail: AtomicUsize::new(0),
            head: AtomicUsize::new(0),
            closed: AtomicBool::new(false),
        }
    }

    /// Producer: push one id. Returns `false` if the ring is full (the consumer is
    /// wedged); the caller decides whether to spin or drop. Non-blocking.
    fn push(&self, id: u32) -> bool {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Acquire);
        if tail.wrapping_sub(head) > self.mask {
            return false; // full
        }
        // SAFETY: SPSC — only the producer writes this slot, and the consumer will
        // not read it until it observes the `Release` store to `tail` below.
        unsafe {
            *self.slots[tail & self.mask].get() = id;
        }
        self.tail.store(tail.wrapping_add(1), Ordering::Release);
        true
    }

    /// Consumer: pop one id, or `None` if empty right now.
    fn pop(&self) -> Option<u32> {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);
        if head == tail {
            return None; // empty
        }
        // SAFETY: the matching `Release` on `tail` above happens-before this read.
        let id = unsafe { *self.slots[head & self.mask].get() };
        self.head.store(head.wrapping_add(1), Ordering::Release);
        Some(id)
    }
}

#[derive(Parser)]
struct Args {
    /// Checkpoint directory (config.json, tokenizer.json, *.safetensors).
    /// Defaults to LFM2_MODEL_DIR, then the LFM2.5-2.6B snapshot in the HF cache.
    #[arg(long)]
    model_dir: Option<PathBuf>,

    /// New-token ceiling. **`0` (the default) means UNBOUNDED** — generation runs
    /// until the model emits EOS (`<|im_end|>`), with no cap, streaming at any
    /// length (the token-ring history wraps, so a long reply no longer bails). A
    /// positive value is a hard cap that stops the run early even before EOS —
    /// use it to bound a measurement or a runaway. (`--ignore-eos` with the
    /// default is refused: nothing would ever stop it.)
    #[arg(long, short = 'n', default_value_t = 0)]
    max_tokens: usize,

    /// Sampling temperature. 0 (the default) is greedy / argmax.
    #[arg(long, short = 't', default_value_t = 0.0)]
    temperature: f64,

    /// Sampling seed (only matters when temperature > 0). If unset, a RANDOM
    /// seed is drawn each run, so repeated `-t > 0` runs explore different
    /// samples; pass an explicit value to reproduce a specific run.
    #[arg(long)]
    seed: Option<u64>,

    /// Force CPU even if a GPU is available.
    #[arg(long)]
    cpu: bool,

    /// How the sampled token feeds the next step and how EOS is learned:
    /// `blocking` (the token is read back every step), `ring` (device-resident
    /// token, one blocking flag read per step), `ring-nonblocking`
    /// (device-resident token, the flag is polled), `ring-hostdrain` (no device
    /// EOS; the host scans each drained window), `ring-gpubatch` (one
    /// `eos_index` dispatch per window, read with a tail wait) or
    /// `ring-gpubatch-nonblocking` (the same index, staged ahead of the
    /// lookahead forward and read only once complete; the host scans
    /// otherwise — the RESULT line's `eos_index=hits/late`). The ring arms need argmax
    /// or plain temperature; top-p is refused on them. **Defaults to
    /// `ring-nonblocking`**: this example is a speed demo and it is the fastest
    /// arm — the token stays on the device AND the EOS flag is polled rather than
    /// read back, so nothing crosses the host boundary per step (§11.5g). It is
    /// the arm that reaches ~60 tok/s steady on M1 Max at B=1.
    #[arg(long, default_value = "ring-nonblocking")]
    readback: String,

    /// Forward passes the host may enqueue before it has learned the previous
    /// step's EOS (ring arms only): 0 or 1. With 1 the GPU always has the next
    /// step queued, and the cache ends one position past an EOS. **Defaults to
    /// 1**: it is the +3.5 % that collects the non-GPU gap (§11.2b), and this
    /// example is single-turn so the one-past-EOS cache ends with the run — the
    /// exact condition `lfm2-release` refuses `--turns > 1` under (§10.2a).
    #[arg(long, default_value_t = 1)]
    lookahead: usize,

    /// How often the decode loop commits and prunes the in-flight queue:
    /// `per-token` or `steps:<N>`. **Defaults to `steps:8`**: the queue bound is
    /// flat at 13–15 across a 7.8× span of n (#520), so committing every 8 steps
    /// never overflows and takes the commit off the per-token path.
    #[arg(long, default_value = "steps:8")]
    commit_budget: String,

    /// How many commits between LIVE completion-marker writes on the
    /// `ring-hostdrain` arm — the wake cadence. `1` writes the marker every commit
    /// (the incumbent per-commit stamp); `N` wakes the host WFE-waiter every N
    /// commits' worth of tokens, so the core gets a longer quiet window to go
    /// dormant between waits (poll.rs: WFE is a bounded spin, not a cheap park, so
    /// a longer wait is what lets it power down). The TTFT ramp still forces the
    /// early commits live, so this cannot delay token 0. Only bites on
    /// `ring-hostdrain` (the other arms stamp per step). **Defaults to 4.**
    #[arg(long, default_value_t = lloom::generation::WAKE_LIVE_EVERY_DEFAULT)]
    wake_cadence: usize,

    /// Drive decode with the **DSpark** speculative drafter instead of the K=1
    /// identity proposer. The drafter proposes a block of up to `--draft` tokens
    /// per round and the target verifies them in one `forward_all`, accepting the
    /// longest argmax-matching prefix plus the target's own bonus token — so the
    /// output is **byte-identical** to the non-speculative path (the target
    /// corrects every mismatch; `design/docs/speculation.md` §10.2n). Off by
    /// default (the current behavior is unchanged). See `--draft-dir` for where
    /// the drafter weights are found.
    ///
    /// At B=1 on an M1 Max this is measured FASTER than the crown non-spec path
    /// (`design/docs/speculation.md` §10.2u: ~80 against ~62 tok/s steady at
    /// 3.5 accepted tokens per round) — once the verify pass took the 16-row
    /// GEMM tile and the ring conv fused; before those it was 0.72×. The round
    /// is still verify-dominated, so the figure moves with the prompt's
    /// acceptance rate; `--phase-sync` shows where a round's time goes.
    #[arg(long)]
    dspark: bool,

    /// Where the DSpark drafter keeps its context KV: `contiguous` (one tensor
    /// per layer, `cat`-grown every round — what every recorded digest belongs
    /// to) or `paged[:<page_size>]` (one page pool per layer read by the paged
    /// `steel_attention`, #708). Sets `LLOOM_DSPARK_CTX_KV`, the switch the
    /// speculator reads at construction, so the two harness modes and the
    /// library agree on one source of truth; unset, the environment decides.
    #[arg(long, value_name = "ARM")]
    dspark_ctx: Option<String>,

    /// The KV ceiling in positions (prompt plus reply). Defaults to the model's
    /// `max_position_embeddings` — 131 072 on LFM2.5-2.6B. The cache does NOT
    /// allocate this: it starts at one `--kv-bucket` and grows a bucket at a
    /// time toward it, each grow admitted against the device first (see
    /// `--kv-grow-fraction`). Passing the ceiling lower is a way to bound a run
    /// by positions rather than by `-n`.
    #[arg(long)]
    kv_capacity: Option<usize>,

    /// How many positions the KV slab grows by when a pass would overrun it.
    /// 4096 (64 MiB at LFM2's 16 KiB/position) is the default and the one every
    /// recorded digest was taken under; a smaller bucket grows more often and
    /// holds less past the reply's end, and emits the same stream — the
    /// `lfm2_kv_growth` gate asserts it. A measurement knob, not an axis.
    #[arg(long, default_value_t = lloom::models::lfm2::DEFAULT_KV_BUCKET)]
    kv_bucket: usize,

    /// The fraction of `recommendedMaxWorkingSetSize` a KV grow may take the
    /// process's Metal allocation to (§9.5c). Defaults to admission's 0.65. A
    /// grow that would cross it is refused with the numbers, the run stops
    /// there, and the tokens already streamed stand — never a kernel panic.
    /// The positive control for that refusal: a value under the weights'
    /// share (~0.10 on this machine) refuses the first grow.
    #[arg(long)]
    kv_grow_fraction: Option<f64>,

    /// Never stop on EOS: the run produces exactly `-n` tokens. The EOS id set
    /// is replaced by an id no tokenizer emits, so every arm's own EOS
    /// mechanism (device flag, host compare, drain scan) still runs and simply
    /// never fires. For long-run gates; the text degenerates past the model's
    /// own stop and is not what a user would read.
    #[arg(long)]
    ignore_eos: bool,

    /// Install the ICB replay executor on the decode path (issue #146/#169). The
    /// stable, packed subset of a decode step is recorded into an
    /// `MTLIndirectCommandBuffer` and replayed each step instead of re-encoded on
    /// the CPU. Forces `--param-style packed` (an ICB command cannot carry an
    /// inline constant). The executor's own overhead was measured +2.28% in the
    /// trace harness (§11.3s), BUT that number was single-turn/single-regime and
    /// never run on this full non-blocking decode — this flag is how we A/B it
    /// here. Off by default. Pair with `--replay-barriers skip-replayed` for the
    /// separable barrier-suppression win (§11.3r, −0.83% GPU-side).
    #[arg(long)]
    replay: bool,

    /// Install the activation arena (`DESIGN.md` §9.2) without the replay
    /// executor: the arm that separates what the arena costs or buys on this
    /// loop from what replay adds on top of it. `--replay` implies it. Recorded
    /// from the first decode steps, as under `--replay`. Not under `--dspark`.
    #[arg(long)]
    arena: bool,

    /// The arena's layout: `packed` (values with disjoint liveness share a
    /// slot — §9.2's 68 KB, and 267 true write-after-read barriers a token
    /// from the reuse, §9.2f) or `non-aliasing` (every value its own slot, no
    /// two ever share bytes — the reference layout, the classical barrier
    /// count exactly, ~3.5 MB). Hazard keying follows the arena: `Range`
    /// unless `CANDLE_METAL_HAZARD_KEY` pins it.
    #[arg(long, default_value = "packed")]
    arena_layout: String,

    /// Route every decode dispatch through its `_packed` entry point and install
    /// the constants pool, WITHOUT the ICB replay executor. The arm that
    /// isolates what moving scalars off `setBytes` (inline, zero-alloc) and into
    /// a pool-bound params buffer costs on the crown loop, separately from what
    /// replay adds on top of it — the same way `--arena` isolates the arena.
    /// The pool makes packed alloc-free (a fresh `new_buffer_with_data` per
    /// dispatch otherwise, §11.3d), so this measures the `setKernelBuffer`-vs-
    /// `setBytes` bind cost at B=1, not allocation. `--replay` implies it (it
    /// needs packed for coverage). Off by default; no perf claim is made for it.
    #[arg(long)]
    packed: bool,

    /// With `--replay`: whether candle's own `auto_barrier` still fires at a
    /// position the ICB already replays. `always` (the shipped behavior — a
    /// redundant barrier at 393 of 505 positions, §11.3p) or `skip-replayed`
    /// (issue #144: suppress candle's barrier at a non-head replayed member,
    /// keeping the ICB's own ordering). `skip-replayed` is the measured −0.83%
    /// GPU-side win (§11.3r) and the reason to run replay at all at B=1.
    #[arg(long, default_value = "skip-replayed")]
    replay_barriers: String,

    /// With `--replay`: how far back an ICB command's barrier scan looks —
    /// `run-start` (#115's shipped behavior: barrier wherever a command conflicts
    /// with anything earlier in the run) or `since-barrier` (scan only to the
    /// previous barrier, matching candle's `auto_barrier`). §11.3l open question 3.
    #[arg(long, default_value = "run-start")]
    replay_barrier_scope: String,

    /// The draft block size K when `--dspark` is set — how many tokens the drafter
    /// proposes per round. The DSpark drafter's block is FIXED at
    /// `block_size − logits_start` (9 for the 2.6B DSpark checkpoint) and has no
    /// runtime cap, so the only accepted value is that ceiling; omit this to use
    /// it. Passing a different K is refused with the reason rather than silently
    /// clamped (a smaller ring would overflow the drafter's fixed block). The flag
    /// is kept so the block size is visible and so a future drafter with a runtime
    /// cap can honor it.
    #[arg(long)]
    draft: Option<usize>,

    /// The DSpark drafter checkpoint directory (config.json + model.safetensors).
    /// Only used with `--dspark`. Defaults to the newest LFM2.5-2.6B-DSpark
    /// snapshot in the HF cache.
    #[arg(long)]
    draft_dir: Option<PathBuf>,

    /// Which cache configuration the NON-speculative path runs on: `crown`
    /// (`Sdpa × Cat × Shuffle`, the fastest B=1 config) or `rewindable` — the
    /// one `--dspark` is forced onto (`Sdpa × InPlace × RotatingRing{K+1}` with
    /// the F32 taps on), run here WITHOUT a drafter, K=1. The pair isolates the
    /// cost of leaving the crown from the cost of the verify loop (#675).
    /// `rewindable-notap` is the same cache with the taps off, so the taps'
    /// own cost separates from the ring's. `--dspark` ignores this and always
    /// runs `rewindable`.
    #[arg(long, default_value = "crown")]
    cache: String,

    /// **Diagnostic, perturbs what it measures**: with `--dspark`, synchronize
    /// the GPU at the end of each phase of a round (draft / verify / observe) so
    /// each phase's wall carries its own GPU tail, and — with
    /// `CANDLE_METAL_PROFILE=1` — take each phase's GPU-busy, dispatch count and
    /// one round's per-kernel census. Off for any throughput figure.
    #[arg(long)]
    phase_sync: bool,

    /// Drop the leading BOS (`<|startoftext|>`, 124894). It is prepended by
    /// DEFAULT: `generation_config.json` and llama.cpp both start LFM2 with it,
    /// and `tokenizers::encode(…, false)` does not, so omitting it ran the model
    /// off its own template. With BOS on (and the EOS set corrected to
    /// `<|im_end|>` only — the two are coupled), DSpark acceptance and throughput
    /// rise on every prompt and replies terminate cleanly: measured
    /// relativity 75.7 → 93.7, structured-extract 139 → 178 tok/s, short-factual
    /// completes on EOS at 90.8. This flag restores the old off-template
    /// tokenization for the A/B only; it lowers acceptance and, without the EOS
    /// fix, drives a non-terminating ramble. Turning BOS on moves the canonical
    /// digests — to the correct, template-conformant output.
    #[arg(long)]
    no_bos: bool,

    /// Override the GEMV+SiLU epilogue fusion (§12.2d) regardless of the cache
    /// arm's baked value. Unset = the arm's default (crown/rewindable bake it
    /// ON). `--gemv-silu false` forces it OFF for the #509 A/B on this harness;
    /// `--gemv-silu true` forces it on. Takes effect after the arm is selected.
    #[arg(long)]
    gemv_silu: Option<bool>,

    /// Override the GEMV residual-add epilogue fusion (§12.2h) regardless of the
    /// cache arm's baked value. Unset = the arm's default (crown/rewindable bake
    /// it ON). `--gemv-residual false` forces it OFF for the #509 A/B.
    #[arg(long)]
    gemv_residual: Option<bool>,

    /// Override the cache preset's attention arm: `sdpa` (the crown), `flash`
    /// (FlashDecoding, the only arm that reads a paged pool) or `generic`. Unset
    /// leaves the preset's. A measurement knob: it changes which kernel the
    /// decode step takes, and the reply's logits move with it (§10.4a) while the
    /// tokens do not on the recorded prompts.
    #[arg(long)]
    attn: Option<String>,

    /// Override the cache preset's KV append arm: `cat`, `in-place` or `paged`
    /// (`paged` needs `--attn flash`; refused otherwise). Unset leaves the
    /// preset's (`in-place` on the crown, `LLOOM_CROWN_KV=cat` the old arm).
    #[arg(long)]
    kv_append: Option<String>,

    // ---- multi-sequence modes (absorbed from the former lfm2-sequence) -------
    /// `single` (default — read one prompt from stdin), `reuse` (a two-turn
    /// conversation decoded with prefix reuse vs a fresh prefill), or
    /// `concurrent` (`--sequences` predefined prompts interleaved on one model,
    /// or pipelined under `--dspark`). The non-single modes print no token text —
    /// they repaint a live raw-stats block and a RESULT line.
    #[arg(long, default_value = "single")]
    mode: String,

    /// Sequences to run under `--mode concurrent`. The prompts are a varied
    /// predefined set of eight (varied so the run exercises prefill), cycled
    /// past eight.
    #[arg(long, default_value_t = 2)]
    sequences: usize,

    /// Under `--mode concurrent`: drive the sequences through the batched-forward
    /// driver (`Session::decode_batched`, #694) — one `m = B` forward per step
    /// for every live sequence, the weight sweep paid once per step — against
    /// each prompt run solo on the selected readback arm. The batched arm takes
    /// `--lookahead` (0: read the ids back before the next forward; 1: feed the
    /// device ids and read back while the next step runs).
    #[arg(long)]
    batched: bool,

    /// Under `--batched`: skip the neighbour-independence arm (the rows re-run
    /// as two separately batched halves, each row asserted equal to its
    /// full-batch stream). On by default; off for a timing-only rotation.
    #[arg(long)]
    no_independence_check: bool,

    /// Under `--batched`: skip the solo arm — a tile A/B at a large `B` wants
    /// only the batched step; the identity columns read `false`.
    #[arg(long)]
    skip_solo: bool,

    /// Tokens per visit under `--mode concurrent` (non-speculative interleave).
    #[arg(long, default_value_t = 16)]
    chunk: usize,

    /// The prefill's spark: how many prompt tokens the FIRST forward takes.
    /// **`0` (default) is the incumbent one-shot** (the whole prompt in one
    /// forward). A positive value runs the prompt as TWO forwards — the spark,
    /// then the whole remainder (not `prompt_len / N` chunks; the weight sweep
    /// is per forward) — through the SAME in-flight pacer the decode loop uses,
    /// so the first token streams off the spark while the bulk runs. It is the
    /// TTFT-vs-throughput dial: a wider spark keeps the matmul units fuller and
    /// reaches one-shot at the prompt length. Honoured by the K=1 loop and by
    /// `--dspark` (the drafter is seeded from each unit as it lands). The
    /// RESULT line reports the arm (`prefill_ubatch=`) and the forwards the run
    /// took (`prefill_units=`). A two-unit run turns on the conv's left-context
    /// carry so it stays byte-identical to one-shot.
    #[arg(long, default_value_t = 0)]
    prefill_ubatch: usize,

    /// The rotating ring's history depth under `--cache rewindable` (non-dspark):
    /// the rewind horizon. `--dspark` sizes it from the drafter's block.
    #[arg(long, default_value_t = 16)]
    ring_k: usize,

    /// Carry the conv state across a turn-boundary prefill (`ConvPrefillCtx`,
    /// `DESIGN.md` §13.2f), for `--mode reuse`. Off by default (§7.1a).
    #[arg(long)]
    conv_prefill_ctx: bool,

    /// Under `--mode concurrent --dspark`: which arm runs first, `solo` or
    /// `pipelined`, so a rotation can alternate the order.
    #[arg(long, default_value = "solo")]
    first: String,

    /// Under `--mode concurrent --dspark`: write every round of both arms as CSV.
    #[arg(long)]
    timeline: Option<PathBuf>,
}

/// The LFM2.5-2.6B snapshot dir in the HF cache, if it is there.
fn default_model_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("LFM2_MODEL_DIR") {
        return Some(PathBuf::from(dir));
    }
    let home = std::env::var("HOME").ok()?;
    let snapshots =
        PathBuf::from(home).join(".cache/huggingface/hub/models--LiquidAI--LFM2.5-2.6B/snapshots");
    // One snapshot hash under there; take the first that has a config.json.
    std::fs::read_dir(&snapshots)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.join("config.json").is_file())
}

/// The safetensors shards for a checkpoint: the sharded index if present, else
/// the single `model.safetensors`.
fn weight_files(dir: &std::path::Path) -> Result<Vec<PathBuf>> {
    let index = dir.join("model.safetensors.index.json");
    if !index.is_file() {
        return Ok(vec![dir.join("model.safetensors")]);
    }
    let json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&index)?).context("reading safetensors index")?;
    let weight_map = json
        .get("weight_map")
        .and_then(|m| m.as_object())
        .context("safetensors index has no weight_map")?;
    let mut names: Vec<String> = weight_map
        .values()
        .filter_map(|v| v.as_str().map(str::to_string))
        .collect();
    names.sort();
    names.dedup();
    Ok(names.into_iter().map(|n| dir.join(n)).collect())
}

/// The DSpark drafter setup resolved before the target config: where its weights
/// live, its constants, and the resolved draft block size `K`.
struct DsparkSetup {
    draft_dir: PathBuf,
    dcfg: DSparkConfig,
    /// The draft block size, capped at the drafter's `max_draft()`.
    k: usize,
}

/// The newest snapshot of a HF repo in the cache with a `config.json`, if any.
/// The same discovery the `lfm2-dspark-verify` example uses for the drafter.
fn newest_snapshot(repo: &str) -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let base = PathBuf::from(home)
        .join(".cache/huggingface/hub")
        .join(repo)
        .join("snapshots");
    let mut entries: Vec<PathBuf> = std::fs::read_dir(&base)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.join("config.json").exists())
        .collect();
    entries.sort();
    entries.pop()
}

/// Read the DSpark drafter's constants from its own `config.json`.
///
/// Never defaulted for the two fields a wrong default silently corrupts —
/// `rope_theta` and `rope_is_neox_style` alter no shape and no dispatch count,
/// and on a drafter there is not even a text gate to catch them because the
/// target verifies every token (`design/docs/speculation.md` §10.2n). Adapted
/// from the `lfm2-dspark-verify` example's `read_drafter_config`.
fn read_drafter_config(dir: &Path) -> Result<DSparkConfig> {
    let raw = std::fs::read_to_string(dir.join("config.json"))
        .with_context(|| format!("reading {}/config.json", dir.display()))?;
    let v: serde_json::Value = serde_json::from_str(&raw)?;
    let dfc = v.get("dflash_config").cloned().unwrap_or_else(|| v.clone());

    let usize_at = |k: &str| -> Result<usize> {
        v.get(k)
            .and_then(|x| x.as_u64())
            .map(|n| n as usize)
            .with_context(|| format!("drafter config has no {k}"))
    };

    let target_layer_ids: Vec<usize> = dfc
        .get("target_layer_ids")
        .and_then(|x| x.as_array())
        .context("drafter config has no dflash_config.target_layer_ids")?
        .iter()
        .filter_map(|x| x.as_u64().map(|n| n as usize))
        .collect();

    Ok(DSparkConfig {
        hidden_size: usize_at("hidden_size")?,
        num_hidden_layers: usize_at("num_hidden_layers")?,
        num_attention_heads: usize_at("num_attention_heads")?,
        num_key_value_heads: usize_at("num_key_value_heads")?,
        head_dim: usize_at("head_dim")?,
        intermediate_size: usize_at("intermediate_size")?,
        vocab_size: usize_at("vocab_size")?,
        rms_norm_eps: v
            .get("rms_norm_eps")
            .and_then(|x| x.as_f64())
            .unwrap_or(1e-5),
        rope_theta: v
            .get("rope_theta")
            .and_then(|x| x.as_f64())
            .context("drafter config has no rope_theta")?,
        rope_is_neox_style: v
            .get("rope_is_neox_style")
            .and_then(|x| x.as_bool())
            .context("drafter config has no rope_is_neox_style")?,
        block_size: usize_at("block_size")?,
        mask_token_id: dfc
            .get("mask_token_id")
            .and_then(|x| x.as_u64())
            .context("drafter config has no dflash_config.mask_token_id")?
            as u32,
        num_target_layers: dfc
            .get("num_target_layers")
            .and_then(|x| x.as_u64())
            .map(|n| n as usize)
            .unwrap_or(30),
        target_layer_ids,
        markov_rank: v
            .get("markov_rank")
            .and_then(|x| x.as_u64())
            .map(|n| n as usize)
            .unwrap_or(0),
        enable_confidence_head: v
            .get("enable_confidence_head")
            .and_then(|x| x.as_bool())
            .unwrap_or(false),
        // NOT a key in config.json — a property of the packaging (§10.2p).
        logits_start: 0,
    })
}

fn main() -> Result<()> {
    let args = Args::parse();

    let model_dir = args
        .model_dir
        .clone()
        .or_else(default_model_dir)
        .context("no --model-dir given and the LFM2.5-2.6B snapshot was not in the HF cache")?;

    // Device: Metal / CUDA / CPU, chosen by what is available (see
    // candle_examples::device). F16 on GPU matches the inference path; CPU has
    // no f16 kernels for everything here, so it runs in f32.
    let device = candle_examples::device(args.cpu)?;
    let dtype = if device.is_cpu() {
        DType::F32
    } else {
        DType::F16
    };

    // ---- load: tokenizer, config, weights --------------------------------
    let load_start = std::time::Instant::now();
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("loading tokenizer: {e}"))?;

    let config: Lfm2Config = serde_json::from_slice(&std::fs::read(model_dir.join("config.json"))?)
        .context("parsing config.json")?;
    let mut config = config.into_config(false);

    // The DSpark drafter's constants, resolved BEFORE the target config because
    // the verify window's width (`K + 1`) sizes the conv ring, and taps must be
    // requested on the target's own layers. Only when `--dspark` is set; the
    // default path never reads a drafter.
    if let Some(arm) = &args.dspark_ctx {
        lloom::models::lfm2_dspark::CtxKv::parse(arm).context("--dspark-ctx")?;
        // SAFETY: single-threaded here, before any drafter or thread exists.
        unsafe { std::env::set_var("LLOOM_DSPARK_CTX_KV", arm) };
    }
    let dspark_setup: Option<DsparkSetup> = if args.dspark {
        let draft_dir = args
            .draft_dir
            .clone()
            .or_else(|| newest_snapshot("models--LiquidAI--LFM2.5-2.6B-DSpark"))
            .context(
                "no LFM2.5-2.6B-DSpark snapshot in the HF cache and no --draft-dir. The drafter is \
                 trained against THIS target (design/docs/speculation.md §10.2l).",
            )?;
        let dcfg = read_drafter_config(&draft_dir)?;
        if dcfg.hidden_size != config.hidden_size {
            anyhow::bail!(
                "drafter hidden {} != target hidden {} — the drafter is not this target's \
                 (design/docs/speculation.md §10.2l)",
                dcfg.hidden_size,
                config.hidden_size
            );
        }
        // The DSpark drafter proposes a FIXED block every round —
        // `block_size − logits_start` tokens (`DSparkSpeculator::draft`, which reads
        // `max_draft()` and does not take a runtime cap). So the verify window is
        // always `max_draft() + 1` positions and the conv ring must reserve that
        // many; a smaller `--draft` cannot make the drafter propose fewer, and
        // sizing the ring below what the drafter emits is the `advance()` overflow
        // this refuses rather than misreports (§7.1a). `--draft` therefore selects
        // the block only when it EQUALS the drafter's own ceiling; anything else is
        // an error with the reason, not a silent clamp.
        let ceiling = dcfg.max_draft();
        let k = match args.draft {
            None => ceiling,
            Some(requested) if requested == ceiling => ceiling,
            Some(requested) => anyhow::bail!(
                "--draft {requested}: the DSpark drafter proposes a fixed block of {ceiling} \
                 tokens (block_size {} − logits_start {}) and has no runtime cap, so only \
                 --draft {ceiling} is supported on this checkpoint. Omit --draft to use the \
                 ceiling.",
                dcfg.block_size,
                dcfg.logits_start,
            ),
        };
        Some(DsparkSetup { draft_dir, dcfg, k })
    } else {
        None
    };

    // The cache arm that runs. `--dspark` needs the REWINDABLE cache, which the
    // crown arms cannot provide: `Cache::advance` requires
    // `ConvState::RotatingRing{ >= K+1 }` (the shuffle is destructive) and
    // `Cache::resolve` requires `KvAppend::InPlace` (the `Cat` arm has no length
    // to rewind) — both refused otherwise (`lfm2.rs`). The verify window is
    // `K + 1` positions (the K proposals plus the target's bonus), so the ring
    // reserves that many history slots. Taps are ON — the drafter's projection
    // consumes the target's layer-output residuals — at F32 (`§10.2m`:
    // deliberate, buys acceptance rate). `Sdpa` attention still applies
    // (token-identical); the GEMV fusions stay on where they compose, and the
    // whole-block conv fusion does not (it is the `Shuffle` arm's, §12.2 row 1).
    // `--cache rewindable` runs the SAME cache without a drafter, so the cost of
    // the cache is measured apart from the cost of the verify loop.
    let cache_arm: &str = if dspark_setup.is_some() {
        "rewindable"
    } else {
        args.cache.as_str()
    };
    if cache_arm == "rewindable" || cache_arm == "rewindable-notap" {
        let k = match &dspark_setup {
            Some(setup) => setup.k,
            None => args.draft.unwrap_or(9),
        };
        config.attn_impl = lloom::models::lfm2::AttnImpl::Sdpa;
        config.kv_append = KvAppend::InPlace;
        config.conv_state = ConvState::RotatingRing { k: k + 1 };
        config.gemv_silu = true;
        config.gemv_residual = true;
        config.conv_fused = true;
        config.tap_dtype = DType::F32;
        config.tap_layers = match (&dspark_setup, cache_arm) {
            (Some(setup), _) => setup.dcfg.tap_layers(),
            (None, "rewindable") => {
                // The 2.6B drafter's taps, `target_layer_ids + 1`, without
                // loading the drafter: the same five layers the `--dspark` run
                // taps, so the arm carries the same per-token capture cost.
                DSparkConfig::lfm2_2_6b().tap_layers()
            }
            _ => Vec::new(),
        };
    } else if cache_arm == "crown" {
        // The fastest B=1 decode config lloom has, baked in — this example exists to
        // show what the engine does at its best, not the conservative defaults every
        // axis ships at (§7.1a keeps those off in the library; a demo turns them on).
        // The crown attention/conv arms (Sdpa × Shuffle, §13.5) plus the two GEMV
        // epilogue fusions (§12.2d/§12.2h) plus the conv fold (§12.2 row 1). Every one
        // is token-identical to the generic path — only the logits reorder — so this
        // is the same answer, faster. The readback strategy (ring + lookahead) is
        // selected below, on the loop side.
        config.attn_impl = lloom::models::lfm2::AttnImpl::Sdpa;
        // KV append: `Cat` is the demo default because it is the fastest arm
        // *today* (measured 2026-09-05: Cat ~60.5 vs InPlace ~58.3 tok/s steady,
        // both byte-identical). InPlace writes in place at a moving offset (stable
        // buffer identity, one `copy2d`/token) and stabilises the attention
        // positions for replay (memory.md §9.2i: 77.9% → 81.7% coverage) — BUT that
        // coverage is latent: no ICB executor is installed on this decode path
        // (`ExecutorSlot::Classical`), so InPlace pays its per-token cost with the
        // replay benefit switched off and loses. InPlace's win is a TWO-part change
        // (InPlace + engage the ICB replay executor); until the second half lands,
        // Cat is faster for this single-turn speed demo. A real serving engine wants
        // InPlace regardless — `Cat` reallocates a wider KV every token and grows
        // unbounded, which does not fly for many concurrent long generations — but
        // that is the serving path's decision, not this demo's. `LLOOM_CROWN_KV=in-place`
        // selects it for the A/B and for the replay work.
        // Flipped: InPlace is now the crown default (was Cat). `LLOOM_CROWN_KV=cat`
        // restores the old arm for the A/B. Being measured: InPlace's per-token
        // cost (one `copy2d`/token/layer for K and V) vs Cat's O(N) realloc+copy
        // — hunting the drain that made InPlace lose (58.3 vs 60.5) despite the
        // better algorithm and the latent replay coverage (§9.2i).
        config.kv_append = match std::env::var("LLOOM_CROWN_KV").as_deref() {
            Ok("cat") => lloom::models::lfm2::KvAppend::Cat,
            _ => lloom::models::lfm2::KvAppend::InPlace,
        };
        config.conv_state = lloom::models::lfm2::ConvState::Shuffle;
        config.gemv_silu = true;
        config.gemv_residual = true;
        config.conv_fused = true;
    } else {
        anyhow::bail!(
            "--cache must be `crown`, `rewindable` or `rewindable-notap`, got `{cache_arm}`"
        );
    }

    // #509 A/B override: the cache arms above bake the GEMV epilogues ON (crown
    // 506-507, rewindable 458-459). These flags force either fusion regardless,
    // so the crown-on vs crown-off delta is measurable on this harness (the
    // library `Config::default` still ships them OFF; this measures the win
    // before flipping that default). Unset leaves the arm's baked value.
    if let Some(v) = args.gemv_silu {
        config.gemv_silu = v;
    }
    if let Some(v) = args.gemv_residual {
        config.gemv_residual = v;
    }
    // The attention and KV-append arms, overriding the preset so the paged pool
    // (which only FlashDecoding reads) and the flash/sdpa pair are measurable on
    // this loop rather than only on the profiler. Applied before `Model::new`
    // and `Cache::new_with`, which both read them.
    if let Some(a) = args.attn.as_deref() {
        config.attn_impl = match a {
            "sdpa" => lloom::models::lfm2::AttnImpl::Sdpa,
            "flash" => lloom::models::lfm2::AttnImpl::FlashDecoding,
            "generic" => lloom::models::lfm2::AttnImpl::Generic,
            other => anyhow::bail!("--attn must be `sdpa`, `flash` or `generic`, got `{other}`"),
        };
    }
    if let Some(kv) = args.kv_append.as_deref() {
        config.kv_append = KvAppend::parse(kv).map_err(|e| anyhow::anyhow!(e))?;
    }
    // `--mode reuse` carries the conv state across the turn-boundary prefill
    // (§13.2f). Set on the config the model is built from; a no-op for `single`.
    //
    // Streamed prefill (`--prefill-ubatch > 0`) REQUIRES it: each chunk after the
    // first is a `seq_len>1` pass onto a warm cache, and without the left-context
    // carry the conv zero-pads the chunk boundary — a different convolution than
    // one pass over the whole prompt, so the output diverges (measured: garbage
    // from char 1). With it on, a chunked prefill computes exactly what one-shot
    // does. A single chunk (or one-shot) is identical either way (cold cache, no
    // left context to carry), so forcing it on when streaming is always safe.
    config.conv_prefill_ctx = args.conv_prefill_ctx || args.prefill_ubatch > 0;

    let files = weight_files(&model_dir)?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, &device)? };
    // `Model::new` is the bulk of the load (~950 ms): it reads the 266 weight
    // tensors through the mmap. Measured config-independent — the crown arms
    // above cost it nothing — so it is the construction touching the mapped
    // bytes, not anything this example selects. A warm page cache hides it (the
    // "sub-second" cold-vs-warm difference); the number the demo shows is the
    // honest steady one.
    // `--replay` needs packed params BEFORE any recorded forward: an ICB command
    // binds one `device const Params*`, not the inline scalars the split arm
    // emits, so the step the executor records must already be packed. Set it here
    // (before warmup) so the recorded shape matches the replayed one.
    // The other two switches replay needs, both process-wide and both before the
    // first pipeline is built: `supportIndirectCommandBuffers` is a pipeline
    // property (§3.7d -- encoding one without it segfaults), and the constants
    // pool is what gives dispatch position N the same params buffer every step
    // (without it every packed position varies and coverage is zero, §11.3d).
    // Packed params + the constants pool are what replay needs for coverage, but
    // they stand on their own: `--packed` installs exactly those two (the crown
    // loop, dispatches routed through `_packed` and bound from the pool) without
    // the ICB executor. `supportIndirectCommandBuffers` is a pipeline property
    // (§3.7d -- encoding an ICB command without it segfaults) that ONLY the
    // replay path needs, so it stays behind `--replay`; packed-without-ICB never
    // encodes an ICB command and never touches it.
    #[cfg(feature = "metal")]
    if args.packed || args.replay {
        use candle::kernels::{set_constants_pool_enabled, set_default_param_style, ParamStyle};
        set_default_param_style(ParamStyle::Packed);
        set_constants_pool_enabled(true);
    }
    #[cfg(feature = "metal")]
    if args.replay {
        candle::kernels::set_pipelines_support_icb(true)
            .map_err(|e| anyhow::anyhow!("enabling supportIndirectCommandBuffers: {e}"))?;
    }

    let model = Model::new(&config, vb).context("constructing LFM2")?;
    // The ceiling is the model's positional ceiling unless the user lowered
    // it; the slab starts at one bucket and grows toward it (`Cache::new_with`).
    let kv_capacity = args.kv_capacity.unwrap_or(config.max_position_embeddings);
    let mut cache = Cache::new_with(true, dtype, &config, &device, kv_capacity)
        .context("allocating KV cache")?
        .with_kv_bucket(args.kv_bucket)
        .context("--kv-bucket")?;
    if let Some(fraction) = args.kv_grow_fraction {
        cache = cache
            .with_kv_grow_fraction(fraction)
            .context("--kv-grow-fraction")?;
    }

    // EOS ids: the config's, plus the chat-end / end-of-text tokens.
    // Stop on `<|im_end|>` (124900) only — the model's declared EOS. `config.json`,
    // `generation_config.json` (`eos_token_id: [124900]`) and the GGUF llama.cpp
    // uses all agree on that single id. `<|endoftext|>` (124895) was ALSO added
    // here, but nothing declares it an EOS — it is the base model's document
    // separator, which the model can emit mid-reply; stopping on it cuts replies
    // early and, in the speculative loop, lets the drafter propose a spurious
    // boundary the target never treats as one. Removed to match the checkpoint's
    // (and llama.cpp's) actual stop set.
    let mut eos_ids: Vec<u32> = config.eos_token_id.into_iter().collect();
    if let Some(id) = tokenizer.token_to_id("<|im_end|>") {
        if !eos_ids.contains(&id) {
            eos_ids.push(id);
        }
    }
    if args.ignore_eos {
        // Contradiction: no EOS AND no token ceiling would never stop. Refuse
        // rather than spin forever (§2.4 — a bad setting bails, not hangs).
        if args.max_tokens == 0 {
            anyhow::bail!(
                "--ignore-eos with -n 0 (unbounded) would never terminate — pass a positive \
                 -n to bound the run when EOS is disabled"
            );
        }
        eos_ids = vec![u32::MAX];
    }

    // Warm the Metal pipelines before the clock the user reads starts. A real
    // forward compiles the prefill GEMM-tile / F32-attention / batched-conv
    // pipelines that the decode step never touches (it routes to GEMV first), so
    // without this they compile *inside* the first prompt's TTFT — ~90 ms of
    // one-time cost `llama.cpp` pays up front and we otherwise pay on the first
    // token. `warmup` runs one prefill + one decode step and resets, so the
    // session is untouched; the cost folds into the load figure below.
    let mut session = Session::new(&model, cache);
    // `LLOOM_SKIP_WARMUP=1` measures the counterfactual: TTFT with the pipelines
    // left cold, so the warmup's effect on the first prompt is attributable
    // rather than asserted. Default is to warm.
    let (libs, pipes) = if std::env::var("LLOOM_SKIP_WARMUP").is_ok() {
        (0, 0)
    } else {
        session.warmup(&device).context("warming Metal pipelines")?
    };

    // Load time, printed before we block on stdin so it is attributable. This is
    // mmap + model construction + cache allocation + pipeline warmup — the
    // safetensors are memory mapped, so the weight *bytes* fault in lazily on the
    // first forward, but the pipelines are now compiled, so the first prompt's
    // TTFT is real prefill work rather than compilation.
    eprintln!(
        "[loaded in \x1b[1;96m{:.2}s\x1b[0m (mmap + construct + warmup: {libs} libs, {pipes} pipelines)]",
        load_start.elapsed().as_secs_f64()
    );

    // The multi-sequence modes (absorbed from lfm2-sequence) branch off here,
    // after the shared load/warmup and before the single path's stdin read and
    // the ICB replay install (which is a single-turn arm). They reuse `model`,
    // `config`, `device` and `eos_ids` — for `--dspark` the config was already
    // forced rewindable+taps above, so the built model is correct for the mode.
    if args.mode != "single" {
        if args.batched && args.dspark {
            anyhow::bail!(
                "--batched drives the K=1 batched-forward body; a batched verify is not built \
                 (design/docs/speculative-pipeline-scheduler-design.md §4b). Pick one of \
                 --batched / --dspark"
            );
        }
        if args.replay || args.arena {
            anyhow::bail!(
                "--replay/--arena are single-turn arms; they do not compose with --mode {} \
                 (the executor records one sequence's step and the concurrent path drives many)",
                args.mode
            );
        }
        let encode = |text: &str| -> Result<Vec<u32>> {
            Ok(tokenizer
                .encode(text, false)
                .map_err(|e| anyhow::anyhow!("tokenizing: {e}"))?
                .get_ids()
                .to_vec())
        };
        let ctx = concurrent::Ctx {
            model: &model,
            config: &config,
            device: &device,
            dtype,
            eos_ids: &eos_ids,
            mode: &args.mode,
            max_tokens: args.max_tokens,
            readback: &args.readback,
            lookahead: args.lookahead,
            commit_budget: &args.commit_budget,
            sequences: args.sequences,
            chunk: args.chunk,
            prefill_ubatch: args.prefill_ubatch,
            dspark: args.dspark,
            batched: args.batched,
            check_independence: !args.no_independence_check,
            skip_solo: args.skip_solo,
            dspark_setup: dspark_setup
                .as_ref()
                .map(|s| (s.draft_dir.clone(), s.dcfg.clone())),
            first: &args.first,
            timeline: args.timeline.clone(),
            encode: &encode,
        };
        return concurrent::run(ctx);
    }

    // Install the ICB replay executor AFTER warmup — so the warmup's forwards do
    // not pollute the recorded step — and before the timed decode. The executor
    // records `ICB_RECORD_STEPS` real decode steps, then replays the stable
    // packed subset each step. `--replay-barriers skip-replayed` is the separable
    // barrier-suppression win (§11.3r); `--replay-barrier-scope` is §11.3l's
    // open question 3. Metal-only; a no-op wherever `--replay` is off.
    // Held so the run can report what the executor actually did: a flag being
    // passed does not establish that anything was replayed (§2.4), and the
    // A/B below is only about the executor if `covered > 0` and `replaying`.
    #[cfg(feature = "metal")]
    let mut icb_exec: Option<std::sync::Arc<candle::kernels::IcbExecutor>> = None;
    // The arena's recording parameters, shared by `--arena` and `--replay`: one
    // step skipped (the first token after the prefill allocates a different
    // set), two recorded, installed at the end of the second.
    #[cfg(feature = "metal")]
    const ARENA_SKIP: usize = 1;
    #[cfg(feature = "metal")]
    const ARENA_RECORD_STEPS: usize = 2;
    #[cfg(feature = "metal")]
    let arena = (args.arena || args.replay) && !args.dspark;
    #[cfg(feature = "metal")]
    let arena_layout = match args.arena_layout.as_str() {
        "packed" => candle::metal_backend::ArenaLayout::Packed,
        "non-aliasing" | "nonaliasing" | "reference" => {
            candle::metal_backend::ArenaLayout::NonAliasing
        }
        other => anyhow::bail!("unknown --arena-layout {other:?}; want packed or non-aliasing"),
    };
    #[cfg(feature = "metal")]
    if arena {
        if let Device::Metal(dev) = &device {
            dev.arm_arena_recording(arena_layout, ARENA_SKIP, ARENA_RECORD_STEPS);
        }
        if !args.replay {
            eprintln!(
                "[arena: {arena_layout:?}, recorded from the first decode steps; no executor]"
            );
        }
    }
    #[cfg(feature = "metal")]
    if args.replay {
        use candle::kernels::metal::ExecutorSlot;
        use candle::kernels::{BarrierScope, IcbExecutor, ReplayBarriers};
        const ICB_RECORD_STEPS: usize = 3;
        let scope = match args.replay_barrier_scope.as_str() {
            "run-start" => BarrierScope::RunStart,
            "since-barrier" => BarrierScope::SinceBarrier,
            other => anyhow::bail!(
                "--replay-barrier-scope must be `run-start` or `since-barrier`, got `{other}`"
            ),
        };
        let barriers = match args.replay_barriers.as_str() {
            "always" => ReplayBarriers::Always,
            "skip-replayed" => ReplayBarriers::SkipReplayed,
            other => anyhow::bail!(
                "--replay-barriers must be `always` or `skip-replayed`, got `{other}`"
            ),
        };
        // `configured` already returns `Arc<IcbExecutor>` — do not re-wrap it
        // (the `Arc<Arc<_>>` is what does not coerce to `Arc<dyn Executor>`).
        let exec = IcbExecutor::configured(ICB_RECORD_STEPS, scope, barriers);
        // The third thing a replay needs after packed params and the pool: stable
        // buffer identity. Under the buffer pool every activation is a different
        // buffer on every step (§9.2c), so every position reads as `varies` and
        // coverage is zero. The activation arena (§9.2) is recorded from the
        // decode loop's own steps -- one skipped, two recorded, installed at the
        // end of the second -- through the brackets `Session::decode` calls. The
        // executor lets the prefill's window, those three steps and one settling
        // step pass before it records; §11.3 says why a recording taken earlier
        // excludes everything for a reason that is not the executor's.
        //
        // Not under `--dspark`: a round's allocation order moves with the
        // accepted length, and the arena serves by ordinal.
        if let Device::Metal(dev) = &device {
            if arena {
                exec.set_warm_in(1 + ARENA_SKIP + ARENA_RECORD_STEPS + 1);
                // After a stale step the arena re-records over the next
                // `ARENA_RECORD_STEPS` steps and installs at the end of the
                // last; the executor records after that and one settling step.
                exec.set_rerecord_warm_in(ARENA_RECORD_STEPS + 1);
            }
            dev.set_executor(std::sync::Arc::new(ExecutorSlot::Custom(exec.clone())));
        }
        icb_exec = Some(exec);
        eprintln!(
            "[replay: ICB executor installed — barriers={barriers:?} scope={scope:?} \
             param-style=packed arena={}]",
            if arena {
                "packed (recorded from the first decode steps)"
            } else {
                "off"
            }
        );
    }

    let sampling = if args.temperature <= 0.0 {
        Sampling::ArgMax
    } else {
        Sampling::All {
            temperature: args.temperature,
        }
    };
    // Random seed when unspecified, so repeated `-t > 0` runs differ; a passed
    // `--seed` reproduces. At temp 0 (greedy/argmax) the seed is irrelevant —
    // the path is deterministic and does not sample.
    let seed = args.seed.unwrap_or_else(rand::random::<u64>);
    let mut logits_processor = LogitsProcessor::from_sampling(seed, sampling);

    // ---- decode: one Session::decode call --------------------------------
    //
    // The strategy is selected here and the loop never names it. Each arm is
    // one value of the `ReadbackStrategy` seam; the loop is generic over it and
    // monomorphizes, so selecting an arm costs nothing per token.
    let budget = CommitBudget::from_str(&args.commit_budget).map_err(anyhow::Error::msg)?;
    if args.phase_sync && !args.dspark {
        anyhow::bail!("--phase-sync brackets the speculative round's phases; it needs --dspark");
    }
    // A speculative round is not a steady-state step: its dispatch structure moves
    // with the accepted length, so every run goes stale within a few rounds. The
    // executor validates a run before issuing it and re-records after a stale
    // step, so the stream is correct either way -- but a plan that is dropped
    // every few rounds covers nothing, and the arena is not armed here at all.
    // Refused so the arm's name is not worn by a classical run (§7.1a-i).
    if args.replay && args.dspark {
        anyhow::bail!(
            "--replay cannot be combined with --dspark: the speculative round's dispatch \
             structure is not stable across rounds, so the executor would re-record every \
             few rounds and replay nothing (DESIGN.md §11.3x)"
        );
    }
    // The in-flight depth: the host/GPU balance warmup measured (`compute_in_flight_depth`),
    // so the pacer is not a static cap — 2 here (GPU step >> host encode), deeper
    // on a faster GPU / smaller model. `LLOOM_IN_FLIGHT` overrides for an A/B.
    let in_flight_depth = match std::env::var("LLOOM_IN_FLIGHT") {
        Ok(v) => v
            .parse()
            .unwrap_or_else(|_| panic!("LLOOM_IN_FLIGHT must be an integer, got {v:?}")),
        Err(_) => session.measured_in_flight_depth(),
    };
    let opts = DecodeOptions {
        max_tokens: args.max_tokens,
        budget,
        lookahead: args.lookahead,
        // Under the speculative body this is the per-phase drain (decode.rs).
        sample_split: args.phase_sync,
        in_flight_depth,
        // 0 = one-shot prefill (incumbent); >0 streams the prompt through the pacer.
        prefill_ubatch: args.prefill_ubatch,
    };

    // The readback strategy is built BEFORE the prompt is read: it depends on
    // nothing in the prompt, and a ring arm allocates its device history and
    // the consumer slack here — tens of milliseconds a serving engine pays once
    // per engine, not once per request, so it is kept off the TTFT clock.
    // `--dspark` drives the speculative body over a `Blocking` handle instead.
    let mut readback_arm = if dspark_setup.is_none() {
        Some(build_readback(
            &args.readback,
            args.wake_cadence,
            &opts,
            &eos_ids,
            &device,
        )?)
    } else {
        // Said once, because the flags are accepted: the speculative body reads
        // its tokens back every round (`verify_tokens`' argmax compare) and
        // commits only under a ring-owning strategy, so none of these reach it
        // and the RESULT line renders them unread (`concurrent::read_or_na`).
        eprintln!(
            "[dspark: --readback {} --lookahead {} --commit-budget {} --wake-cadence {} are not \
             read by the speculative body; the RESULT line renders them n/a(dspark)]",
            args.readback, args.lookahead, args.commit_budget, args.wake_cadence
        );
        None
    };

    // ---- prompt: read stdin, wrap in the LFM2 chat template ---------------
    let mut prompt = String::new();
    std::io::stdin()
        .read_to_string(&mut prompt)
        .context("reading prompt from stdin")?;
    let prompt = prompt.trim();
    let text = format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n");
    let mut prompt_ids = tokenizer
        .encode(text.as_str(), false)
        .map_err(|e| anyhow::anyhow!("tokenizing prompt: {e}"))?
        .get_ids()
        .to_vec();
    // BOS by default — the model's template start (see `--no-bos`). Coupled with
    // the `<|im_end|>`-only EOS set above: together they lift acceptance and make
    // replies terminate; BOS without the EOS fix drove a non-terminating ramble.
    if !args.no_bos {
        let bos = tokenizer
            .token_to_id("<|startoftext|>")
            .context("tokenizer has no <|startoftext|> — LFM2's BOS")?;
        prompt_ids.insert(0, bos);
    }

    // The DSpark proposer, built only when `--dspark` is set. It borrows the
    // target's token embedding and `lm_head` (the drafter ships neither — 62
    // tensors, `lfm2_dspark.rs`) as cheap `Arc`-backed clones. The RoPE table is
    // sized for the whole generation, context positions included.
    let mut dspark_spec: Option<DSparkSpeculator> = match &dspark_setup {
        None => None,
        Some(setup) => {
            let draft_file = setup.draft_dir.join("model.safetensors");
            if !draft_file.exists() {
                anyhow::bail!("{} does not exist", draft_file.display());
            }
            // Cover the largest absolute position the draft will attend: the prompt,
            // every accepted token, plus the block width and slack. `max_tokens`
            // is the token ceiling — EXCEPT `0` means UNBOUNDED (run until EOS),
            // where it is NOT the bound: the run is bounded in practice by the KV
            // ceiling (`kv_capacity`, the model's positional ceiling unless
            // `--kv-capacity` lowered it), which is the furthest the target — and
            // so the drafter, which attends the same positions — can reach. Sizing
            // the drafter's RoPE table from `max_tokens=0` gave it just
            // `prompt+block+64` rows (89 for a 16-token prompt), so an unbounded
            // `--dspark` run crashed the moment it drafted past ~position 82
            // (`DSparkRotary::apply`'s `cos.narrow`). Use the KV ceiling instead.
            let generation_ceiling = if args.max_tokens == 0 {
                kv_capacity
            } else {
                args.max_tokens
            };
            let max_pos = prompt_ids.len() + generation_ceiling + setup.dcfg.block_size + 64;
            // F16 to match the target the block is embedded with — the drafter's
            // stored BF16 has no CPU matmul and would need a cast at every meeting
            // of the two dtypes (`lfm2_dspark.rs` `DSparkDrafter::new` docs).
            let dvb =
                unsafe { VarBuilder::from_mmaped_safetensors(&[draft_file], DType::F16, &device)? };
            let drafter = DSparkDrafter::new(setup.dcfg.clone(), max_pos, dvb)
                .context("constructing DSpark drafter")?;
            eprintln!(
                "[dspark: drafter loaded, {} layers, draft block K={} (verify window {} positions, \
                 conv ring reserves {} history slots)]",
                setup.dcfg.num_hidden_layers,
                setup.k,
                setup.k + 1,
                setup.k + 1
            );
            let spec = DSparkSpeculator::new(
                drafter,
                model.draft_embedding().clone(),
                model.lm_head_linear().clone(),
                1,
            )
            .context("constructing DSpark speculator")?;
            eprintln!("[dspark: context kv {}]", spec.ctx_kv().name());
            Some(spec)
        }
    };

    let mut spec = IdentityProposer;

    // ---- decode: stream tokens to stdout as they are produced -------------
    //
    // The callback fires with each committed batch (ring) or each token
    // (blocking) and prints incrementally, so a reply appears as it is written
    // rather than after EOS. `TokenOutputStream` handles the BPE boundary — a
    // token that lands mid-multibyte yields nothing until the next completes it,
    // so the printed text is never broken UTF-8. That the tok/s below holds under
    // this is the point: the drain reads finished work off the ring, it does not
    // wait, so streaming does not re-serialize the non-blocking path.
    use std::io::Write;
    // The profile window is the decode: warmup's and the load's command
    // buffers are dropped so `gpu_busy / decode_wall` below is one window over
    // one window. Warmup ended on a `synchronize`, so nothing is in flight.
    #[cfg(feature = "metal")]
    if candle::kernels::metal::profile::enabled() {
        candle::kernels::metal::profile::reset();
    }
    // The allocator-side counters at the start of the clock, so the run's own
    // churn prints as deltas: pool misses (`allocations`), evictions, residency
    // adds/removes/commits, and the readbacks split fast / blit / null-after-wait.
    // A mechanism whose engagement nobody counts is one whose null reads as a
    // result (§2.4); these existed and nothing printed them.
    #[cfg(feature = "metal")]
    let churn_before = match &device {
        Device::Metal(m) => Some(churn_snapshot(m)),
        _ => None,
    };
    let decode_start = std::time::Instant::now();
    // Page faults the process takes during the decode (`getrusage`, two calls
    // off the loop): the number that says whether there is anything left to
    // pin. Every device buffer is in the residency set already (§6.3b); this
    // counts the host side -- heap, stack, text, the CPU-mapped ring pages.
    let faults_before = rusage_faults();
    // Wall-clock to the first token the consumer is HANDED (`first_batch_at`, on
    // the decode thread when it pushes) and to the first VISIBLE token
    // (`first_visible_ns`, on the CONSUMER thread when a byte reaches stdout).
    // Measurement seam; the visible stamp is shared because it is set off-thread.
    let mut first_batch_at: Option<std::time::Duration> = None;
    let first_visible_ns = Arc::new(AtomicU64::new(0));
    // The two hops between `received` and `visible`, on the consumer thread:
    // the first id popped off the SPSC, and the first id detokenized (lloom
    // #715: the 32 ms between the two headline stamps had no owner).
    let first_pop_ns = Arc::new(AtomicU64::new(0));
    let first_detok_ns = Arc::new(AtomicU64::new(0));
    // Every stdout write's stamp, for the visible cadence (how bursty the
    // stream is at the terminal): one `u64` push per write, off the hot thread.
    let visible_stamps: Arc<std::sync::Mutex<Vec<u64>>> = Arc::new(std::sync::Mutex::new(
        Vec::with_capacity(spsc_cap_hint(opts.max_tokens)),
    ));
    // Device memory at the first token the consumer is handed -- i.e. after the
    // prefill and the first step -- so a cut-period sweep can read what the
    // prefill's intermediates cost in `pending` (§13.5o's open question).
    let mut device_allocated_at_first_token: Option<usize> = None;

    // The output decouple: the decode loop pushes ids into this SPSC (non-blocking,
    // no syscall on the hot thread) and a separate consumer thread detokenizes and
    // writes stdout. Sized to cover the whole run so `push` never blocks in
    // practice. `-n 0` (unbounded) gets a generous 1M-slot ring.
    let spsc_cap = if opts.max_tokens == 0 {
        1 << 20
    } else {
        opts.max_tokens + 1024
    };
    let spsc = Arc::new(TokenSpsc::new(spsc_cap));

    // The consumer thread: pop → detokenize → write stdout. A slow terminal stalls
    // HERE, never the decode loop. Owns the tokenizer, the `TokenOutputStream` and
    // the stdout handle for the whole run; parks briefly when the ring is empty
    // (no notify from the producer — that per-commit futex wake is what spiked
    // `step_waits` when this was an mpsc channel).
    let consumer = {
        let spsc = Arc::clone(&spsc);
        let first_visible_ns = Arc::clone(&first_visible_ns);
        let first_pop_ns = Arc::clone(&first_pop_ns);
        let first_detok_ns = Arc::clone(&first_detok_ns);
        let visible_stamps = Arc::clone(&visible_stamps);
        std::thread::spawn(move || {
            let mut tos = candle_examples::token_output_stream::TokenOutputStream::new(tokenizer);
            let stdout = std::io::stdout();
            let mut out = stdout.lock();
            let stamp = |slot: &AtomicU64| {
                slot.compare_exchange(
                    0,
                    decode_start.elapsed().as_nanos() as u64,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                )
                .ok();
            };
            loop {
                match spsc.pop() {
                    Some(id) => {
                        stamp(&first_pop_ns);
                        let text = tos.next_token(id);
                        stamp(&first_detok_ns);
                        if let Ok(Some(text)) = text {
                            stamp(&first_visible_ns);
                            // Direct write to the locked stdout; a write that the
                            // kernel cannot take right now blocks THIS thread only.
                            let _ = out.write_all(text.as_bytes());
                            let _ = out.flush();
                            if let Ok(mut v) = visible_stamps.lock() {
                                v.push(decode_start.elapsed().as_nanos() as u64);
                            }
                        }
                    }
                    None => {
                        if spsc.closed.load(Ordering::Acquire) {
                            break;
                        }
                        // Nothing right now: yield the core briefly rather than spin
                        // a poll. ~200 µs is invisible at a terminal cadence and
                        // costs the producer nothing.
                        std::thread::sleep(std::time::Duration::from_micros(200));
                    }
                }
            }
            // The tail the stream still holds (a partial multibyte run), then flush.
            if let Ok(Some(rest)) = tos.decode_rest() {
                let _ = out.write_all(rest.as_bytes());
            }
            let _ = out.flush();
        })
    };

    let out = {
        // The hot-thread callback: push each drained id into the SPSC. Atomic
        // stores only — no detokenize, no `print!`, no `flush`, no allocation, no
        // notify. `push` returns false only if the ring is full (a wedged
        // consumer), in which case spin until it drains — the one place a
        // sustained-backpressure stall is allowed to land, and unreachable for a
        // ring sized to the run.
        let mut on_tokens = |batch: &[u32]| -> candle::Result<()> {
            if first_batch_at.is_none() && !batch.is_empty() {
                first_batch_at = Some(decode_start.elapsed());
                #[cfg(feature = "metal")]
                if let Device::Metal(m) = &device {
                    device_allocated_at_first_token = Some(m.device().current_allocated_size());
                }
            }
            for &id in batch {
                while !spsc.push(id) {
                    std::hint::spin_loop();
                }
            }
            Ok(())
        };
        match dspark_spec.as_mut() {
            Some(dspark) => {
                // The speculative body. It does NOT ride the device-resident ring —
                // the accepted tokens are host-side after each verify pass's argmax
                // compare (`design/docs/decode-driver-design.md`: the speculative
                // path's one readback per round is inherent) — so it streams the
                // accepted block straight to `on_tokens` and learns EOS host-side.
                // A `Blocking` strategy supplies the device handle and commit
                // cadence; its own feed/EOS paths are unused here.
                let mut rb = Blocking::new(&eos_ids, &device)?;
                session.decode_speculative(
                    dspark,
                    &mut rb,
                    &mut logits_processor,
                    &prompt_ids,
                    &opts,
                    &eos_ids,
                    &mut on_tokens,
                )?
            }
            None => decode_arm(
                readback_arm
                    .as_mut()
                    .expect("a non-speculative run built its readback arm above"),
                &mut session,
                &mut spec,
                &mut logits_processor,
                &prompt_ids,
                &opts,
                &mut on_tokens,
            )?,
        }
    };
    let decode_elapsed = decode_start.elapsed();
    let faults_after = rusage_faults();
    // Signal the consumer no more ids are coming and wait for it to drain what
    // remains, print the multibyte tail, and flush.
    spsc.closed.store(true, Ordering::Release);
    consumer.join().ok();
    let first_visible_at = match first_visible_ns.load(Ordering::Relaxed) {
        0 => None,
        ns => Some(std::time::Duration::from_nanos(ns)),
    };

    let out_ids = out.ids;
    let n = out_ids.len();
    let end_to_end = n as f64 / decode_elapsed.as_secs_f64();

    println!();
    std::io::stdout().flush().ok();

    let st = &out.stats;

    // Two figures, because they answer different questions (§2.5.3):
    //  * end-to-end includes the prompt's prefill, which is a different shape of
    //    work and drifts with prompt length — what a user of a one-shot waits.
    //  * steady is the median decode step, prefill excluded — what the engine
    //    sustains token-to-token, and the number to compare against the §320
    //    floor. The median, not the mean, because one scheduling hiccup (or a
    //    loaded machine) drags the mean and says nothing about the steady path.
    let mut steps: Vec<u64> = st.step_ns.clone();
    let steady = if steps.is_empty() {
        // RELEASE (no per-step `step_ns` — the clock reads are compiled out of the
        // hot path): the coarse llama-bench figure, measured host-side off the loop
        // — generated tokens ÷ (decode wall-clock − prefill). No per-token
        // `mach_absolute_time()` touched the steady path to produce it, so it cannot
        // be skewed by the measurement itself. A debug/`--features profile` build
        // takes the per-step median branch below instead.
        let decode_only = decode_elapsed.as_secs_f64() - (st.prefill_ns as f64 / 1e9);
        if decode_only > 0.0 {
            n as f64 / decode_only
        } else {
            f64::NAN
        }
    } else {
        // Median is taken on a sorted copy; the unsorted `step_ns[0]` above is the
        // first step, so sort AFTER reading the TTFT term.
        steps.sort_unstable();
        let med_ns = steps[steps.len() / 2] as f64;
        1e9 / med_ns
    };
    // TTFT = prompt submitted → first token available = the prefill forward plus
    // the first decode step (the first token is the prefill's own argmax, fed and
    // sampled in step 0). This is where the mmap'd weights fault in, so a cold
    // run's TTFT carries the load the `[loaded …]` line above deferred.
    // TTFT — the headline, made fair against llama-bench GIVEN our async design.
    //
    // llama-bench times a SYNCHRONOUS `llama_decode` (it blocks on the GPU) around
    // the prompt and calls the elapsed span TTFT. We must NOT introduce that sync —
    // the non-blocking pipeline is the whole design. The fair analog is the moment
    // the host FIRST RECEIVES the first token, observed NON-BLOCKING on the hot
    // thread and BEFORE any detokenize/print: `first_batch_at`. It captures the
    // same span llama-bench's `-p` does (prefill + first-token compute, the GPU
    // produced it and the host saw it) and excludes the same things (sampling,
    // detokenize, print) — but without a forced sync, which is the honest
    // architectural difference. The consumer-thread "to-screen" time is reported
    // separately below (`visible`), it is NOT the headline.
    //
    // A debug/`--features profile` build additionally has the per-step `step_ns`
    // and prints the finer compute term (`prefill + step0`) instead.
    // Three stamps, named for what they measure. `submit` is the host's encode
    // of the prompt plus the first argmax with NO wait on the GPU — a host
    // submit time, not a time-to-token, and not what llama-bench's `pp` times
    // (that call blocks on the prompt's execution). `received` is when the
    // first token exists on the host: the strategy's own setup, the prefill's
    // execution and the first drain — the TTFT, and what the RESULT line
    // carries. `visible` adds the consumer thread and the tokenizer's boundary
    // rule (a first piece that is not alphanumeric-terminated waits for the
    // next token, which is one decode step).
    let submit_ms = st.ttft_compute_ns as f64 / 1e6;
    let ttft_ms = first_batch_at.map_or(f64::NAN, |d| d.as_secs_f64() * 1e3);
    let ms_or_na = |ns: u64| {
        if ns == 0 {
            "n/a".to_string()
        } else {
            format!("{:.2}", ns as f64 / 1e6)
        }
    };
    eprintln!(
        "[first-token: submit {:.2} ms (host encode, no GPU wait) | received {} ms (TTFT) | popped {} ms | detokenized {} ms | visible {} ms (to stdout); tail after the loop {:.2} ms]",
        submit_ms,
        first_batch_at
            .map(|d| format!("{:.2}", d.as_secs_f64() * 1e3))
            .unwrap_or_else(|| "n/a".into()),
        ms_or_na(first_pop_ns.load(Ordering::Relaxed)),
        ms_or_na(first_detok_ns.load(Ordering::Relaxed)),
        first_visible_at
            .map(|d| format!("{:.2}", d.as_secs_f64() * 1e3))
            .unwrap_or_else(|| "n/a".into()),
        st.tail_ns as f64 / 1e6,
    );
    // The visible cadence: the gaps between consecutive stdout writes. A
    // bursty stream (a commit budget of N steps, a wide cut period) shows up
    // as a high p99 against a p50 near the step; the count says how many
    // writes carried the run's tokens.
    if let Ok(stamps) = visible_stamps.lock() {
        if stamps.len() >= 2 {
            let mut gaps: Vec<f64> = stamps
                .windows(2)
                .map(|w| (w[1] - w[0]) as f64 / 1e6)
                .collect();
            gaps.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let q = |f: f64| gaps[((gaps.len() - 1) as f64 * f) as usize];
            let burst_1ms = gaps.iter().filter(|g| **g < 1.0).count();
            eprintln!(
                "[visible cadence: {} writes over {} tokens; gap p50 {:.2} / p90 {:.2} / p99 {:.2} / max {:.2} ms; {} gaps < 1 ms (tokens arriving in a burst)]",
                stamps.len(),
                n,
                q(0.5),
                q(0.9),
                q(0.99),
                gaps[gaps.len() - 1],
                burst_1ms
            );
        }
    }
    if let Some(bytes) = device_allocated_at_first_token {
        eprintln!(
            "[prefill: device_allocated {:.3} GB at the first token (after the prefill and the first step)]",
            bytes as f64 / 1e9
        );
    }
    // The two TTFT terms apart, and the first steps by phase: `prefill_ns` is
    // the prompt's encode, and under `lookahead = 1` step 0's EOS wait is where
    // the prefill's GPU tail lands, so the split says which side a first-token
    // cost sits on.
    let ms = |ns: u64| ns as f64 / 1e6;
    // The hot thread's scheduling class for the run (`generation::HotThread`,
    // claimed inside `Session::decode*`) and the faults the process took
    // meanwhile: engagement, not a flag (§2.4).
    {
        let h = &st.hot_thread;
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
            faults_after.0.saturating_sub(faults_before.0),
            faults_after.1.saturating_sub(faults_before.1),
        );
    }
    // The per-step split is a debug/profile detail — `step_ns` is empty in a plain
    // release build (the reads are compiled out), so print it only when populated.
    if !st.step_ns.is_empty() {
        eprintln!(
            "[ttft split: prefill {:.2} ms + step0 {:.2} ms; step_ns[1..={}] = {}; prompt_tokens={}]",
            ms(st.prefill_ns),
            ms(st.step_ns.first().copied().unwrap_or(0)),
            st.step_ns.len().min(5).saturating_sub(1),
            st.step_ns
                .iter()
                .skip(1)
                .take(4)
                .map(|&n| format!("{:.2}", ms(n)))
                .collect::<Vec<_>>()
                .join(" "),
            prompt_ids.len(),
        );
    }
    for (name, census) in [
        ("prefill", &st.prefill_census),
        ("step 0", &st.step0_census),
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
    for (i, p) in st.first_steps.iter().enumerate() {
        eprintln!(
            "[step {i} phases: sample {:.2} ms, forward {:.2} ms, backpressure {:.2} ms, eos_wait {:.2} ms]",
            ms(p.sample_ns),
            ms(p.forward_ns),
            ms(p.backpressure_ns),
            ms(p.eos_ns)
        );
    }
    // Bold bright-cyan on the figures worth eyeballing (the reset closes it), so
    // they stand out when scanning a terminal. `\x1b[1;96m` = bold + bright cyan,
    // `\x1b[0m` = reset. Only the numbers are wrapped; the labels stay plain.
    let b = |s: String| format!("\x1b[1;96m{s}\x1b[0m");

    // Under `--dspark` a "step" is a speculative ROUND, not a token: a round
    // publishes 1..K+1 tokens (the accepted prefix plus the target's bonus). So
    // `steady` above is rounds/s, and the token throughput is that times the mean
    // tokens per round. Both are reported honestly rather than relabeling one as
    // the other. `n` counts the streamed anchor plus every round's accepted
    // tokens; `st.steps` counts the rounds after the anchor.
    if args.dspark {
        let rounds = st.steps.max(1);
        // Tokens the ROUNDS produced (the whole stream minus the prefill anchor
        // that is emitted before round 0).
        let round_tokens = n.saturating_sub(if n > 0 { 1 } else { 0 });
        let tokens_per_round = round_tokens as f64 / rounds as f64;
        let steady_tok_s = steady * tokens_per_round; // rounds/s × tokens/round
        let setup = dspark_setup.as_ref().expect("--dspark implies a setup");
        eprintln!(
            "[dspark: {} accepted-tokens/round (mean over {} rounds), \
             {} tok/s steady (rounds/s × tokens/round), K={}]",
            b(format!("{tokens_per_round:.2}")),
            rounds,
            b(format!("{steady_tok_s:.1}")),
            setup.k,
        );
        report_rounds(st, args.phase_sync);
    }

    // The KV-append arm within the cache config, so `cache=crown` is not
    // ambiguous between the `Cat` and `InPlace` arms (they print the same
    // `cache_arm` string otherwise; the crown default is `InPlace` since the
    // A/B inverted, `LLOOM_CROWN_KV=cat` restores `Cat`).
    let kv_arm = match config.kv_append {
        KvAppend::InPlace => "inplace",
        KvAppend::Cat => "cat",
        KvAppend::Paged => "paged",
    };
    let attn_arm = match config.attn_impl {
        lloom::models::lfm2::AttnImpl::Sdpa => "sdpa",
        lloom::models::lfm2::AttnImpl::FlashDecoding => "flash",
        lloom::models::lfm2::AttnImpl::Generic => "generic",
    };
    // The KV slab's state and the process's device allocation, so a run reports
    // the memory it actually held at the length it reached rather than the
    // ceiling it could have (§9.5's admission is a prediction; this is the
    // reading).
    {
        let kv = session.sequence().cache().kv_report();
        let mib = |b: usize| b as f64 / (1024.0 * 1024.0);
        #[cfg(feature = "metal")]
        let device_line = match &device {
            Device::Metal(m) => format!(
                " device_allocated={:.2} GB working_set={:.2} GB",
                m.device().current_allocated_size() as f64 / 1e9,
                m.device().recommended_max_working_set_size() as f64 / 1e9
            ),
            _ => String::new(),
        };
        #[cfg(not(feature = "metal"))]
        let device_line = String::new();
        #[cfg(feature = "metal")]
        if let (Device::Metal(m), Some(before)) = (&device, churn_before.as_ref()) {
            let after = churn_snapshot(m);
            eprintln!(
                "[churn over the run: pool lookups={} hits={} allocations={} evicted={} \
                 trims={} | residency adds={} removes={} commits={} | to_cpu fast={} blit={} \
                 null_after_wait={} staged={}]",
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
            );
        }
        eprintln!(
            "[kv: len={} slab={} ceiling={} bucket={} grows={} kv_bytes={:.1} MiB{device_line}]",
            kv.kv_len,
            kv.slab,
            kv.capacity,
            kv.bucket,
            kv.grows,
            mib(kv.bytes)
        );
    }
    eprintln!(
        "[{n} tokens: {} tok/s end-to-end (prefill included), \
         {} {} , TTFT {} ms, \
         {} on {:?}; cache={cache_arm} attn={attn_arm} kv={kv_arm} bos={} \
         readback={} lookahead={} prefill_ubatch={} prefill_units={} \
         commit_budget={} commits={} max_in_flight={} \
         step_markers={} step_waits={} cut={} pipelined_cut={} \
         ring_publishes={} poll_fallbacks={} eos_index={}/{} overrun={} eos={} param-style={}]",
        b(format!("{end_to_end:.1}")),
        b(format!("{steady:.1}")),
        if args.dspark {
            "rounds/s steady (median round); see the dspark line above for tok/s"
        } else if st.step_ns.is_empty() {
            // Release: coarse tokens ÷ (wall − prefill), measured host-side off the
            // loop (llama-bench style); no per-step clock read skews it.
            "tok/s steady (wall-clock, decode only)"
        } else {
            "tok/s steady (median decode step)"
        },
        b(format!("{ttft_ms:.1}")),
        if dtype == DType::F16 { "f16" } else { "f32" },
        device,
        !args.no_bos,
        if args.dspark {
            "dspark"
        } else {
            &args.readback
        },
        concurrent::read_or_na(!args.dspark, args.lookahead),
        args.prefill_ubatch,
        st.prefill_units,
        concurrent::read_or_na(!args.dspark, budget.arm()),
        st.commits,
        st.max_in_flight,
        st.step_markers,
        st.step_waits,
        cut_policy_arm(&device),
        concurrent::read_or_na(!args.dspark, pipelined_cut_arm(&device)),
        st.readback.ring_publishes,
        st.readback.poll_fallbacks,
        st.readback.eos_index_hits,
        st.readback.eos_index_late,
        st.overrun,
        out.eos,
        param_style_arm(),
    );
    #[cfg(feature = "metal")]
    if let Some(exec) = icb_exec.as_ref() {
        report_replay(exec, st.steps);
    }
    // `--packed` without `--replay`: no ICB executor to report, but the pool is
    // live and its arms are what say whether packed stayed alloc-free on this
    // loop (`detached`/`rewritten` past warm-up mean a position took a private
    // buffer instead of a pooled slot — the alloc packed was meant to avoid).
    #[cfg(feature = "metal")]
    if args.packed && !args.replay {
        let (slots, c) = candle::kernels::constants_pool()
            .map(|p| (p.len(), p.counters()))
            .unwrap_or_default();
        eprintln!(
            "[packed: constants_pool slots={slots} allocated={} unchanged={} rewritten={} \
             detached={}]",
            c.allocated, c.unchanged, c.rewritten, c.detached,
        );
    }
    // GPU busy per step, when the census is on: the split between an arm
    // costing the host and an arm costing the GPU (§11.3s landed GPU-side).
    // `drain_all` above synchronised, so every completion handler has run.
    #[cfg(feature = "metal")]
    if candle::kernels::metal::profile::enabled() {
        let s = candle::kernels::metal::profile::snapshot();
        let steps = st.steps.max(1) as f64;
        // GPU time by kernel, from each command buffer's last dispatch label.
        // Exact only at CANDLE_METAL_COMPUTE_PER_BUFFER=1 (one dispatch per
        // buffer -- the per-kernel arm, lloom #715/#716); at the default period
        // a buffer is charged to its last kernel, and the `buffers` column
        // (≈ dispatches/50) says so.
        {
            let by = candle::kernels::metal::profile::gpu_time_by_label();
            let total: f64 = by.iter().map(|(_, _, t)| *t).sum();
            eprintln!(
                "[profile: gpu time by kernel ({} labels, {:.3} ms/step summed over {:.0} buffers/step; exact iff one dispatch per buffer):",
                by.len(),
                1e3 * total / steps,
                by.iter().map(|(_, n, _)| *n as f64).sum::<f64>() / steps
            );
            for (label, buffers, secs) in by.iter().take(24) {
                eprintln!(
                    "    {:>8.3} ms/step  {:>6.1} buf/step  {:>7.1} µs/buf  {}",
                    1e3 * secs / steps,
                    *buffers as f64 / steps,
                    1e6 * secs / (*buffers as f64).max(1.0),
                    label
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
            st.steps,
        );
        // The falsifier of issue #702, on one window: the union of the decode's
        // command-buffer GPU intervals against the decode's wall (prefill on both
        // sides). `1 − busy/wall` is the time no command buffer was executing —
        // an upper bound on GPU idle, since a buffer stalled on a fence inside
        // its interval still counts as busy.
        let wall = decode_elapsed.as_secs_f64();
        eprintln!(
            "[profile: gpu_busy_union {:.4} s / decode_wall {:.4} s = {:.4} busy/wall; \
             gpu_span {:.4} s; gpu_busy_sum {:.4} s]",
            s.gpu_busy_union_s,
            wall,
            s.gpu_busy_union_s / wall.max(1e-9),
            s.gpu_span_s,
            s.gpu_busy_sum_s,
        );
        // The command-buffer lifecycle, priced on the hot thread: what
        // `commit_swap_locked` (encoder close, residency, signal, handlers,
        // commit, next-buffer creation, swap) and `flush_and_prune` cost the
        // encoding thread per step.
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
        report_idle_gaps(&s, st, wall);
    }
    // The hot thread's steady-state split, every step summed (debug/profile
    // builds only): how much of a step the host spends encoding against how
    // much it spends waiting for the GPU to publish the token.
    if st.phase_steps > 0 {
        let n = st.phase_steps as f64;
        let p = &st.phase_totals;
        let total = (p.sample_ns + p.forward_ns + p.backpressure_ns + p.eos_ns) as f64;
        eprintln!(
            "[hot thread per step, mean over {} steps: sample {:.3} ms, forward (encode) {:.3} ms, \
             backpressure {:.3} ms, eos_wait {:.3} ms; encoding {:.1} % of the bracketed step]",
            st.phase_steps,
            ms(p.sample_ns) / n,
            ms(p.forward_ns) / n,
            ms(p.backpressure_ns) / n,
            ms(p.eos_ns) / n,
            100.0 * (p.sample_ns + p.forward_ns) as f64 / total.max(1.0),
        );
    }
    // The per-step host time's SHAPE (debug/profile builds, where `step_ns` is
    // populated): the median is the RESULT line's `steady`; the mean, the tail
    // percentiles and `mean/median` say how far the run is from one number, and
    // the dump below puts every step beside its GPU idle so a slow step can be
    // read against whether the GPU was idle in it (issue #702 reopened).
    if st.step_ns.len() > 1 {
        let mut v: Vec<f64> = st
            .step_ns
            .iter()
            .skip(1)
            .map(|&ns| ns as f64 / 1e6)
            .collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        eprintln!(
            "[step time, host, steps 2..={} (the first step carries the prefill's wait): p50 {:.3} / mean {:.3} / p90 {:.3} / \
             p99 {:.3} / max {:.3} ms; mean/median {:.4}; steps over 1.5× the median: {}]",
            v.len() + 1,
            quantile(&v, 0.5),
            mean,
            quantile(&v, 0.9),
            quantile(&v, 0.99),
            v.last().copied().unwrap_or(f64::NAN),
            mean / quantile(&v, 0.5).max(1e-9),
            v.iter().filter(|&&x| x > 1.5 * quantile(&v, 0.5)).count(),
        );
    }
    #[cfg(feature = "metal")]
    if arena {
        if let Device::Metal(dev) = &device {
            match dev.arena_auto_report() {
                // The key is what the arena's cost rests on (§9.2e), so the
                // line that says the arena engaged says which key it ran under.
                Some(r) => eprintln!(
                    "[arena: {:?} installed after decode step {}: {} of {} ordinals served; \
                     hazard_key={:?} rerecordings={} counters={:?}]",
                    r.layout,
                    r.installed_at_step,
                    r.covered,
                    r.total,
                    candle::metal_backend::hazard_key(),
                    dev.arena_rerecordings(),
                    dev.arena_counters(),
                ),
                None => eprintln!("[arena: not installed]"),
            }
        }
    }
    Ok(())
}

/// The process's (minor, major) page-fault counts so far, from `getrusage`;
/// zeros where it is unavailable.
/// How many stdout writes a run can take: one per token at most.
fn spsc_cap_hint(max_tokens: usize) -> usize {
    if max_tokens == 0 {
        1 << 16
    } else {
        max_tokens + 16
    }
}

fn rusage_faults() -> (u64, u64) {
    #[cfg(unix)]
    {
        // SAFETY: `rusage` is plain data; zeroed is a valid value, and the
        // pointer passed is to that live local.
        let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
        if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) } == 0 {
            return (ru.ru_minflt as u64, ru.ru_majflt as u64);
        }
    }
    (0, 0)
}

/// `p`-quantile of an ascending `v` by nearest rank; NaN when empty.
fn quantile(v: &[f64], p: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v[(((v.len() - 1) as f64) * p).round() as usize]
}

/// The GPU's idle between consecutive command buffers, as a distribution and
/// attributed -- the instrument issue #702's reopening asked for. The union
/// reports `1 − busy/wall` as one sum over the run; this says whether that
/// idle is spread evenly (the seam between every pair of buffers, GPU-side)
/// or concentrated in a few steps (a transient stall), and for each gap
/// whether the next buffer was committed **after** the previous one ended
/// (the host starved the GPU) or was already queued (the driver's seam,
/// which no host thread can shorten).
///
/// `CANDLE_METAL_PROFILE_DUMP=<path>` writes the raw series: every command
/// buffer (`<path>`: step, commit, start, end) and every step's idle beside
/// its host step time (`<path>.steps.csv`), for the plot.
#[cfg(feature = "metal")]
fn report_idle_gaps(
    s: &candle::kernels::metal::profile::Snapshot,
    st: &lloom::nn::DecodeStats,
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
    // Histogram of the gaps, count and ms per bucket.
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
    // Per step: the idle laid at the step whose buffer ended it. Step 0 is
    // everything committed before the loop's first step (the prefill).
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
    if let Ok(path) = std::env::var("CANDLE_METAL_PROFILE_DUMP") {
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

/// The command-buffer cut policy in force, read back from the device rather
/// than from the environment that selected it (§7.1a-i), with the counter's
/// period beside it.
/// The period a pipelined step loop raises the cut to
/// (`Commands::pipelined_compute_per_buffer`), beside `cut=`: since lloom #715
/// `cut=` names the default the prefill and a blocking loop run at, and a
/// `lookahead ≥ 1` loop runs its steps at this one.
fn pipelined_cut_arm(device: &Device) -> String {
    #[cfg(feature = "metal")]
    if let Device::Metal(dev) = device {
        return dev.pipelined_compute_per_buffer().to_string();
    }
    let _ = device;
    "n/a".to_string()
}

fn cut_policy_arm(device: &Device) -> String {
    #[cfg(feature = "metal")]
    if let Device::Metal(dev) = device {
        return candle::metal_backend::CutPolicy::render(
            dev.cut_policy(),
            dev.compute_per_buffer(),
        );
    }
    let _ = device;
    "n/a".to_string()
}

/// The param style the decode path actually ran, read from the process-wide
/// default rather than from the flag that set it (§7.1a-i): `--replay` and
/// `--packed` both select `packed`, everything else is `split` (inline
/// `setBytes`). Reported so a run's summary line names which bind path produced
/// its number.
fn param_style_arm() -> &'static str {
    #[cfg(feature = "metal")]
    {
        use candle::kernels::ParamStyle;
        return match candle::kernels::default_param_style() {
            ParamStyle::Packed => "packed",
            ParamStyle::Split => "split",
        };
    }
    #[cfg(not(feature = "metal"))]
    "split"
}

/// What the ICB executor did on this run: engagement first, then the residency
/// term (`DESIGN.md` §11.3w). `replaying=false` or `covered=0` means the A/B
/// above compared the classical path against itself wearing this arm's name.
#[cfg(feature = "metal")]
fn report_replay(exec: &candle::kernels::IcbExecutor, steps: u64) {
    use candle::kernels::metal::trace;
    let cov = exec.coverage();
    eprintln!(
        "[replay: replaying={} covered={}/{} runs={} barriers_encoded={} stale={} \
         stale_runs={} prefix_flushes={} rerecordings={} discarded_windows={} \
         excluded_varies={} excluded_inline_consts={} excluded_no_pipeline={}]",
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
    );
    eprintln!("[replay: warm_in_skipped={}]", exec.warm_in_skipped());
    let ranges = trace::execute_range_count();
    let use_resource = trace::use_resource_count();
    let steps = steps.max(1) as f64;
    // The residency arm is named from the plan, not from the variable that
    // selected it (§7.1a-i): per-run declares `residency_per_step`, plan-wide
    // declares `runs x plan_wide`.
    let arm = if exec.runs() > 0 && exec.residency_per_step() == exec.runs() * exec.plan_resident()
    {
        "plan-wide"
    } else {
        "per-run"
    };
    eprintln!(
        "[replay: execute_ranges={ranges} ({:.1}/step) useResource={use_resource} ({:.1}/step) \
         residency={arm} per_step={} plan_wide={} barriers_suppressed={}]",
        ranges as f64 / steps,
        use_resource as f64 / steps,
        exec.residency_per_step(),
        exec.plan_resident(),
        trace::barriers_suppressed(),
    );
    // The pool's four arms (`ConstantsPool`): `detached > 0` is the check that
    // keeps a lookahead loop from rewriting a params block under an in-flight
    // reader having fired; `rewritten` past the first steps means a position's
    // block changes between steps and cannot be replayed from its slot.
    let pool = candle::kernels::constants_pool()
        .map(|p| (p.len(), p.counters()))
        .unwrap_or_default();
    eprintln!(
        "[replay: constants_pool slots={} allocated={} unchanged={} rewritten={} detached={}]",
        pool.0, pool.1.allocated, pool.1.unchanged, pool.1.rewritten, pool.1.detached,
    );
    for ((kernel, reason), n) in &cov.excluded_by_kernel {
        eprintln!("[replay excluded {n:4} {kernel:<44} {reason}]");
    }
    // What went stale, attributed: which axis of the recording moved, on which
    // kernels, and the first occurrences dated by step -- a count says a run
    // stopped matching, this says what moved and when.
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

/// The per-round breakdown of a `--dspark` run: where a round's wall goes, per
/// round and per emitted token, from `DecodeStats::rounds`.
///
/// The first `WARMUP_ROUNDS` are dropped from every figure — the first pass
/// through the drafter's kernels pays pipeline construction and a cold pool —
/// and the dropped count is printed. Wall figures are medians per round and
/// ratios of sums per token (§10.2j's denominator rule); GPU-busy and dispatch
/// figures appear only under `--phase-sync` with `CANDLE_METAL_PROFILE=1`, and
/// the census is one round's, the one the loop kept.
fn report_rounds(st: &lloom::nn::DecodeStats, phase_sync: bool) {
    const WARMUP_ROUNDS: usize = 4;
    let rounds = &st.rounds;
    if rounds.len() <= WARMUP_ROUNDS {
        eprintln!(
            "[dspark rounds: only {} rounds, fewer than the {WARMUP_ROUNDS} dropped as warm-up; \
             no breakdown]",
            rounds.len()
        );
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
    eprintln!(
        "[dspark rounds: {} timed ({WARMUP_ROUNDS} warm-up dropped), {accepted} tokens accepted, \
         {tokens_per_round:.3} tokens/round; phase_sync={phase_sync}]",
        timed.len()
    );
    eprintln!("[dspark per round, ms (median | mean):  draft {med_draft:.3} | {:.3}   verify {med_verify:.3} | {:.3}   observe {med_observe:.3} | {:.3}   round {med_round:.3} | {:.3}]",
        sum_draft / n, sum_verify / n, sum_observe / n, sum_round / n);
    eprintln!("[dspark per accepted token, ms (ratio of sums):  draft {:.3}   verify {:.3}   observe {:.3}   total {:.3}]",
        sum_draft / e, sum_verify / e, sum_observe / e, sum_round / e);
    // The accepted-length distribution: the round cost is paid whatever the
    // acceptance, so this is the other factor in the per-token figure.
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
    // Host readbacks per phase: each is a wait for the GPU, and the round owes
    // exactly one (the verify's argmax). Counted, not inferred — a draft that
    // drained its ids to the host shows here and nowhere in the digests.
    let rb_draft: u64 = timed.iter().map(|r| r.timing.draft_readbacks).sum();
    let rb_verify: u64 = timed.iter().map(|r| r.timing.verify_readbacks).sum();
    let rb_observe: u64 = timed.iter().map(|r| r.timing.observe_readbacks).sum();
    eprintln!(
        "[dspark host readbacks per round (mean):  draft {:.2}   verify {:.2}   observe {:.2}   round {:.2}]",
        rb_draft as f64 / n,
        rb_verify as f64 / n,
        rb_observe as f64 / n,
        (rb_draft + rb_verify + rb_observe) as f64 / n
    );
    let gpu_draft: f64 = timed.iter().map(|r| r.timing.draft_gpu_s * 1e3).sum();
    let gpu_verify: f64 = timed.iter().map(|r| r.timing.verify_gpu_s * 1e3).sum();
    let gpu_observe: f64 = timed.iter().map(|r| r.timing.observe_gpu_s * 1e3).sum();
    let disp_draft: u64 = timed.iter().map(|r| r.timing.draft_disp).sum();
    let disp_verify: u64 = timed.iter().map(|r| r.timing.verify_disp).sum();
    let disp_observe: u64 = timed.iter().map(|r| r.timing.observe_disp).sum();
    let profiled = disp_draft + disp_verify + disp_observe > 0;
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

/// The allocator-side counters a run moves, summed over the shared and the
/// private pool, read at the two ends of the decode clock and printed as deltas.
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
}

#[cfg(feature = "metal")]
fn churn_snapshot(m: &candle::MetalDevice) -> ChurnSnapshot {
    let (shared, private) = m.pool_counters();
    let res = m.residency_counters();
    let (fast, blit) = candle::metal_backend::to_cpu_counts();
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
        to_cpu_null: candle::metal_backend::to_cpu_null_after_wait(),
        to_cpu_staged: candle::metal_backend::to_cpu_staged_count(),
    }
}

/// One readback strategy, selected by name and held across the prompt read.
enum ReadbackArm {
    Blocking(Blocking),
    Ring(Ring, &'static str),
}

/// Build the selected strategy. An unknown name is an error rather than a
/// fallback (a run that reports one arm and executes another is the failure
/// this refuses).
fn build_readback(
    readback: &str,
    wake_cadence: usize,
    opts: &DecodeOptions,
    eos_ids: &[u32],
    device: &Device,
) -> Result<ReadbackArm> {
    // The device-side id history, sized EXACTLY from the per-step geometry — a
    // wrapping ring whose live window is `period × quantum`, NOT the whole
    // generation, so an unbounded (`-n 0`) run streams at any length by reusing
    // the ring. B=1 greedy here (IdentityProposer): quantum = 1, so the ring is
    // tens of slots, always CPU-mapped, zero-copy. (`Geometry::cap` is the tight,
    // no-margin size; a lap is a loud bail, not silent corruption.)
    let geometry = lloom::generation::Geometry {
        batch: 1,
        max_draft: 0,
        period: opts.budget.period(),
        lookahead: opts.lookahead,
        // Widen the ring's input buffer to the streamed-prefill chunk so a chunk
        // stages into one reused buffer; 0 leaves it at `quantum` (the one-shot
        // path allocates its whole-prompt input tensor via the trait default, so
        // the input buffer width is not consulted there).
        max_input_width: opts.prefill_ubatch(),
        // The input ring needs `depth + 1` buffers so a host-staged chunk does not
        // race an in-flight forward's read; the pacer depth is what bounds that.
        in_flight_depth: opts.pacer_depth(),
    };
    let built = std::time::Instant::now();
    let arm = match readback {
        "blocking" => ReadbackArm::Blocking(Blocking::new(eos_ids, device)?),
        "ring" => ReadbackArm::Ring(
            Ring::with_geometry(eos_ids, device, geometry, EosRead::Blocking)?,
            "ring",
        ),
        "ring-nonblocking" => ReadbackArm::Ring(
            Ring::with_geometry(
                eos_ids,
                device,
                geometry,
                EosRead::Poll(PollMechanism::default()),
            )?,
            "ring-nonblocking",
        ),
        // The device never computes the EOS flag (~5 dispatches/token gone);
        // the host learns EOS from the drained ids on the commit cadence, so
        // up to a commit-worth of forwards may run past the EOS and their
        // tokens are truncated from the output. Byte-identical to the poll
        // arm — the truncation lands at the same position the poll would
        // break at. The wake-cadence knob only bites on the deferred-marker
        // arms: `--wake-cadence N` wakes the host every N commits instead of
        // every one (the other arms stamp per step).
        "ring-hostdrain" => ReadbackArm::Ring(
            Ring::with_geometry(eos_ids, device, geometry, EosRead::HostOnDrain)?
                .with_wake_cadence(wake_cadence),
            "ring-hostdrain",
        ),
        // Like ring-hostdrain, but EOS is learned by ONE batched `eos_index` GPU
        // dispatch per window instead of a host scan — the host reads back a
        // single index per commit. Byte-identical to the other arms.
        "ring-gpubatch" => ReadbackArm::Ring(
            Ring::with_geometry(eos_ids, device, geometry, EosRead::GpuBatched)?
                .with_wake_cadence(wake_cadence),
            "ring-gpubatch",
        ),
        // ring-gpubatch without its per-window tail wait: the index is staged
        // ahead of the lookahead forward and read only once its own buffer has
        // completed; the host scans when it has not. `eos_index=hits/late` on
        // the RESULT line says which answered.
        "ring-gpubatch-nonblocking" => ReadbackArm::Ring(
            Ring::with_geometry(eos_ids, device, geometry, EosRead::GpuBatchedNonblocking)?
                .with_wake_cadence(wake_cadence),
            "ring-gpubatch-nonblocking",
        ),
        other => anyhow::bail!(
            "--readback must be `blocking`, `ring`, `ring-nonblocking`, `ring-hostdrain`, \
             `ring-gpubatch` or `ring-gpubatch-nonblocking`, got `{other}`"
        ),
    };
    // Stamped because a ring arm allocates its history and the consumer slack
    // here; the figure is what a serving engine pays once per engine.
    eprintln!(
        "[readback: {readback} built in {:.2} ms (LLOOM_RING_SLACK_BYTES={})]",
        built.elapsed().as_secs_f64() * 1e3,
        std::env::var("LLOOM_RING_SLACK_BYTES").unwrap_or_else(|_| "default".into()),
    );
    Ok(arm)
}

/// Run the loop over the built strategy. The loop is generic and this is where
/// the type is chosen, one call site per arm.
fn decode_arm(
    arm: &mut ReadbackArm,
    session: &mut Session<'_, Model>,
    spec: &mut IdentityProposer,
    sampler: &mut LogitsProcessor,
    prompt_ids: &[u32],
    opts: &DecodeOptions,
    on_tokens: &mut dyn FnMut(&[u32]) -> candle::Result<()>,
) -> Result<DecodeOut> {
    let out = match arm {
        ReadbackArm::Blocking(rb) => {
            session.decode(spec, rb, sampler, prompt_ids, opts, on_tokens)?
        }
        ReadbackArm::Ring(rb, name) => {
            let out = session.decode(spec, rb, sampler, prompt_ids, opts, on_tokens)?;
            // The arm that ran is the arm reported: a ring whose poll could not
            // read the mapped words fell back to the blocking read (or the
            // synchronising snapshot at each drain), and says so rather than
            // passing as non-blocking.
            if !rb.poll_is_mapped() {
                eprintln!(
                    "[{name}: the ring's words were not CPU-mapped; the EOS read took the \
                     blocking fallback ({} of {} steps) and each drain the synchronising \
                     snapshot]",
                    out.stats.readback.poll_fallbacks, out.stats.steps
                );
            }
            debug_assert!(*name == "ring" || rb.arms().nonblocking_eos);
            out
        }
    };
    Ok(out)
}
