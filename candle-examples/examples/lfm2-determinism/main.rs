//! Is LFM2 decode bit-deterministic on Apple silicon, including under GPU load?
//!
//! Measurement harness for lloom issue #5 / `DESIGN.md` §2.3.8, §16 P0 #5.
//!
//! The question is whether the *hardware* is deterministic: same dispatch, same
//! inputs, same bits, regardless of occupancy, thermal state and concurrent GPU
//! work. Asking it of the real model rather than a synthetic reduction kernel
//! exercises the actual dispatch mix, the actual allocator behaviour and the
//! actual fence pattern, which is where nondeterminism would plausibly come
//! from.
//!
//! Two digests are reported per run, and the distinction matters:
//!
//! * **tokens** — SHA-256 over the generated token ids. This is what the issue
//!   asks for, but it is the *weaker* signal: sampling quantizes logits to a
//!   token, so a bit-level difference usually maps to the same token and is
//!   invisible. It only becomes visible when a difference happens to straddle
//!   a sampling boundary.
//! * **logits** — SHA-256 over the raw little-endian f32 bits of every logit
//!   vector at every step, folded in step order. This sees a single flipped
//!   mantissa bit at any of the 128000 * n_steps values, whether or not it ever
//!   changes a token.
//!
//! A logits digest that is stable across runs is a far stronger claim than a
//! stable token stream, so both are printed and both are compared.
//!
//! # This harness reports no throughput, deliberately
//!
//! `lfm2-decode-profile` is the one that measures throughput, and its
//! `wall_ms_per_token` is a steady-state per-token mean with warmup excluded
//! under argmax. Nothing here is comparable to it: this runs multi-turn with
//! sampling on, counts prefill and warmup, and lets `kv_len` grow across turns.
//!
//! The `elapsed_ms` below is a duration and not a rate. **Do not divide it by
//! `generated`** — that reconstructs the whole-generation average
//! `measurements/issue-7-reconciliation.md` §1 exists to discredit, and it is
//! what the removed `tok_per_s` field was (lloom #102).
//!
//! The reason it cannot be fixed by relabelling is structural: two arms of a
//! determinism A/B generate different numbers of tokens *by construction*,
//! because when the digests diverge the arms take different paths and stop at
//! different points. So `elapsed_ms` is not comparable between arms either, and
//! neither is anything derived from it.
//!
//! ```bash
//! cargo run --release --example lfm2-determinism -- \
//!   --model-dir /path/to/LFM2.5-VL-3B --n 500
//! ```

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

mod sha256;

use anyhow::{Context, Result};
use candle::metal_backend::ArenaLayout;
use candle::{DType, Device, IndexOp, Tensor};
use candle_examples::lfm2_setup::{default_model_dir, parse_config, tensor_names, weight_files};
use clap::Parser;
use lloom::generation::{LogitsProcessor, Sampling};
use lloom::models::lfm2::{AttnImpl, Cache, ConvState, KvAppend, Model};
use lloom::nn::ops::KvQuant;
use lloom::nn::VarBuilder;
use std::path::PathBuf;
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(about = "LFM2 decode determinism probe")]
struct Args {
    /// Local checkpoint directory (config.json, tokenizer.json, weights).
    #[arg(long)]
    model_dir: Option<PathBuf>,

    /// Prompt fed to the model. Fixed across runs by default.
    #[arg(
        long,
        default_value = "Explain, in careful detail, how a modern operating system schedules threads across CPU cores, and why fairness and throughput are in tension."
    )]
    prompt: String,

    /// Number of tokens to generate.
    #[arg(long, short = 'n', default_value_t = 500)]
    n: usize,

    /// Sampling seed.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// Sampling temperature. 0 selects argmax.
    #[arg(long, default_value_t = 0.7)]
    temperature: f64,

    /// Nucleus sampling cutoff.
    #[arg(long, default_value_t = 0.9)]
    top_p: f64,

    /// Run on the CPU instead of Metal.
    #[arg(long)]
    cpu: bool,

    /// Weight dtype. Defaults to f16 on Metal, matching how ambrogio loads it.
    #[arg(long)]
    dtype: Option<String>,

    /// Feed the prompt verbatim instead of wrapping it in the chat template.
    ///
    /// Without the template LFM2 emits EOS at the first step and then degenerates,
    /// which measures nothing; the wrapped form is what ambrogio actually sends.
    #[arg(long)]
    raw_prompt: bool,

    /// Emit every token id and its per-step logits digest.
    #[arg(long)]
    dump_tokens: bool,

    /// Write every step's raw f32 logits to this path, as
    /// `step <i> <v0> <v1> ...` one line per step.
    ///
    /// **A digest cannot answer "does the error grow".** §2.3.5a's second
    /// discriminator is that a reduction-order difference is ulp-scale and does
    /// *not* grow with reduction length — which is a statement about magnitude,
    /// and a SHA-256 destroys magnitude by construction: two logit vectors one
    /// ulp apart and two that are unrelated both give unequal 64-hex strings.
    /// This dumps the values so the difference between two arms can be
    /// subtracted rather than merely observed to exist (issue #141, §10.2g).
    #[arg(long)]
    dump_logits: Option<String>,

    /// Decode this fixed token sequence instead of the sampled one, so two arms
    /// walk identical positions and their `--dump-logits` output can be
    /// subtracted (issue #170).
    ///
    /// **`--dump-logits` is half an instrument on its own, and this is the other
    /// half.** §2.3.5a's second discriminator — *is the error ulp-scale, and
    /// does it grow?* — is a comparison of two logit vectors **at the same
    /// position**. But a determinism A/B's arms generate different tokens by
    /// construction: when the digests diverge the arms take different paths and
    /// stop at different points (§2.4, and #97's 267 against 401). From the
    /// first divergent step onward the two dumps describe different
    /// conversations, so subtracting them measures the prompts drifting apart
    /// rather than the arithmetic differing. §10.2g could subtract its dumps
    /// *only* because that axis happened not to diverge in tokens — the generic
    /// arm's stream was byte-identical while the logits digest moved — and an
    /// axis whose arms diverge is exactly the case worth investigating.
    ///
    /// Forcing pins the sequence: every arm is driven through the same tokens,
    /// so position `i` means the same thing in every dump and `max|Δlogit|` at
    /// `i` is a property of the arithmetic alone.
    ///
    /// Format is one id per line, `#` comments and blank lines ignored; a bare
    /// whitespace- or comma-separated list on one line also parses, so the
    /// `TOKENS` line this harness prints can be fed back in directly.
    ///
    /// **What is still sampled, and why it must be.** The logits are produced,
    /// digested and dumped exactly as they would be; the sampler still runs, and
    /// its choice is *recorded and then discarded* in favour of the forced id.
    /// That divergence count is the interesting output rather than a diagnostic:
    /// it is how many positions the arm would have left the reference path at,
    /// which is what the token digest conflates with everything downstream of it.
    #[arg(long)]
    force_tokens: Option<String>,

    /// Decode speculatively with a `K`-position verify pass (issue #89).
    ///
    /// **This is the generic verifier and it carries no proposer.** Where the K
    /// tokens come from is policy and belongs to #90/#91 (`DESIGN.md` §7.7:
    /// policy is runtime-swappable, kernel structure is not). What this flag
    /// supplies is the *mechanism* — propose K, verify in one pass, accept a
    /// prefix, roll back K−n — driven by whichever proposer `--spec-proposer`
    /// names, both of which are sequences rather than models.
    ///
    /// Requires `--conv-state rotating:<K'>` with `K' >= K` and
    /// `--kv-append in-place`; `Cache::advance` refuses everything else before
    /// any state is written, and says why.
    ///
    /// **Under greedy decoding the output must be identical to the
    /// non-speculative run**, so the canonical digest pair is this mechanism's
    /// test rather than a divergence bound — a scheme that changes the output is
    /// a bug and not a tradeoff.
    #[arg(long)]
    speculate: Option<usize>,

    /// Which proposer drives `--speculate`. Both are sequences; neither is a
    /// model.
    ///
    /// * `oracle` — always proposes what the target would have produced, taken
    ///   from `--force-tokens`. **The transparency arm**: acceptance is total,
    ///   so the digests must be unmoved, which is #89's first acceptance item.
    /// * `wrong:<N>` — the oracle with every `N`-th proposal corrupted. **The
    ///   rollback arm**: acceptance is partial, the rollback fires on every
    ///   window, and the digests must *still* be unmoved. That is the stronger
    ///   of the two, because a mechanism that is transparent only when nothing
    ///   is rejected has not been shown to roll back at all.
    #[arg(long, default_value = "oracle")]
    spec_proposer: String,

    /// Label printed with the result, to tag a run inside a batch.
    #[arg(long, default_value = "")]
    label: String,

    /// Verify the bundled SHA-256 against known vectors, then exit.
    #[arg(long)]
    self_test: bool,

    /// Ignore EOS and generate the full token budget.
    ///
    /// Note that past its natural stopping point this checkpoint falls into a
    /// two-token cycle, which is a weak test: a self-reinforcing attractor
    /// would likely reproduce even under real nondeterminism. Prefer
    /// `--turns` for long streams.
    #[arg(long)]
    ignore_eos: bool,

    /// Number of conversational turns to run against one KV cache.
    ///
    /// `DESIGN.md` §2.3.6 calls long multi-turn the strongest determinism test:
    /// beyond amplifying bit differences it exercises KV reuse, a growing and
    /// non-monotone `kv_len`, and changing chunk counts — the paths where
    /// order-dependence would actually live. Each turn appends a follow-up
    /// prompt to the same cache, so the stream stays on-distribution instead of
    /// degenerating the way `--ignore-eos` does.
    #[arg(long, default_value_t = 1)]
    turns: usize,

    /// Serve decode activations from an activation arena (`DESIGN.md` §9.2).
    ///
    /// Two steady-state decode steps are recorded to derive the plan, the arena
    /// is installed, and everything after runs against it.
    #[arg(long)]
    arena: bool,

    /// Serve the arena's session-state declines from an owned region (#47,
    /// `DESIGN.md` §9.2x).
    ///
    /// **This flag is why the digest gate matters for #47.** The region hands
    /// out views of one owned allocation at planned offsets, so a wrong offset
    /// aliases two session tenants — and §9.3 says nothing would report it. A
    /// session tenant is read by the *next* step, so the corruption would appear
    /// one token after the write that caused it, which is precisely the shape
    /// §2.3.2 says looks like benign float noise. The digest is the only
    /// available detector, and an **unmoved** pair is the required result rather
    /// than a pleasant one: the region changes which buffer answers, never what
    /// is computed.
    ///
    /// Requires `--arena`, and has a population only under `--kv-append cat`.
    #[arg(long)]
    session_region: bool,

    /// Arena layout: `packed` or `non-aliasing` (§9.3's reference).
    #[arg(long, default_value = "non-aliasing")]
    arena_layout: String,

    /// Where arena offsets come from: `cpu` (#69) or `gpu` (issue #70).
    ///
    /// **This flag is why the gate exists.** The GPU allocator's per-step reset
    /// is ordered by a device-scope fence plus candle's barrier, and §3.5 says a
    /// wrongly-ordered reset is silent corruption rather than an error -- no
    /// barrier count can detect it, so the digest is the only available detector
    /// (§2.3.2). A single forward pass will not do: §15.1 #7 requires N runs of
    /// a long multi-turn generation, which additionally exercises KV reuse and a
    /// growing `kv_len`.
    #[arg(long, default_value = "cpu")]
    arena_offsets: String,

    /// Which attention implementation decode takes: `generic` (the default, and
    /// the arm that must still reproduce the digests recorded from §8.1b through
    /// §11.3k) or `sdpa` (the GQA-native `sdpa_vector` kernel, issue #97).
    ///
    /// The `sdpa` arm is *expected* to produce a different pair, for the reasons
    /// recorded in `measurements/issue-97-prediction.md` before it was run:
    /// online softmax rescales the accumulator per key, `fast::exp` is not
    /// candle's `exp`, and the score is a `simd_sum`. What it must not do is
    /// produce a pair that *varies between runs* -- the kernel walks keys by
    /// index with no float atomics and no completion-order merge, so §2.3.3 #1
    /// holds by construction and a varying digest would mean something else is
    /// wrong.
    ///
    /// **Carried here beside the arena flags so a combination can be gated.**
    /// `bench/issue-70-det` had the arena arms and no `--attn`;
    /// `bench/issue-97-det` had `--attn` and no arena. Neither could gate
    /// `Sdpa × arena`, which is why #105's C6 reports counts rather than digests.
    #[arg(long, default_value = "generic")]
    attn: String,

    /// KV tokens per page for `--attn flash` — the **allocation** granularity.
    ///
    /// §10.4 proposes 256 and marks it **UNVERIFIED**. A flag rather than a
    /// constant because §10.3d establishes page size as two dispatch-tier
    /// numbers, so 16, 256 and 1024 are one field apart — which is what
    /// *"must not foreclose it"* requires. Verifying which wins is #61's axis.
    #[arg(long, default_value_t = 256)]
    flash_page_size: usize,

    /// Pages per chunk for `--attn flash` — the `k` of
    /// `chunk_size = k * page_size`.
    ///
    /// §10.4 fixes the page and the chunk equal **by fiat**; §9.1d records that
    /// a page (allocation) and a tile (computation) are optimised against
    /// disjoint cost functions, so **a sweep holding `k = 1` cannot separate a
    /// page-size effect from a tile-size one**. 1 is what §10.4 specifies; the
    /// flag is what makes the two separable.
    #[arg(long, default_value_t = 1)]
    flash_k: usize,

    /// How `--attn flash` sizes the scratch class: `grow` (the default),
    /// `reserve` or `bucket` — `DESIGN.md` §9.1a's three policies, issue #234.
    ///
    /// Sizing decides **where bytes live and not what they are**, so all three
    /// arms owe the same text here and the same digests in
    /// `lfm2-determinism` — which is why this flag exists on the gates rather
    /// than only on the profiler. `grow` is what #116 allocated, so the default
    /// is byte-for-byte what shipped.
    #[arg(long, default_value = "grow")]
    flash_scratch_sizing: String,

    /// KV cache growth: `cat` (the default `Tensor::cat` reallocation) or
    /// `in-place` (pre-allocated, written at a moving offset -- issue #142).
    ///
    /// **The digests must not move across this axis**, which is what makes it
    /// different from `--attn`. `Sdpa` changes the arithmetic and legitimately
    /// moves them (§2.3.8c); `in-place` changes only *where the bytes live*, so
    /// a moved digest here is a defect rather than a variant (§2.3.5a). Gate it
    /// against the pair for whichever `--attn` arm is selected, not against the
    /// canonical generic pair.
    #[arg(long, default_value = "cat")]
    kv_append: String,

    /// KV cache element width: `off` (the f16 default), `int8` or `int4`
    /// (issue #218, `DESIGN.md` §10.7).
    ///
    /// **A moved digest is EXPECTED on this axis and is not itself the
    /// finding** -- quantization is not a reordering, it is a real precision
    /// loss, so §2.3.5a's dichotomy does not apply unchanged and §2.3.3c's
    /// tier-2 sub-distinction does: what is owed is the **magnitude** of the
    /// ulp delta and its **growth slope**, against the ladder in §10.7a
    /// (FlashDecoding 1.50, `RotatingRing` 2.75, `MathMode` 3.50 f16 ulp,
    /// against #128's known defect at 1840.06). Magnitude is the primary
    /// reading: #128's defect ratio is 0.787, so growth alone would have called
    /// a real defect benign.
    #[arg(long, default_value = "off")]
    kv_quant: String,

    /// How decode writes conv state: `shuffle` (the default, §6.1's
    /// `narrow` + `Tensor::cat`), `ring[:K[:slack]]` (the sliding window,
    /// §10.2e) or `rotating[:K]` (§10.2a's rotating index, §10.2g).
    ///
    /// **The two ring arms differ in what this harness should expect, and the
    /// difference is the point.** `ring` slides, so its live window is
    /// `l_cache` contiguous slots in the shuffle's own order: the summation
    /// order is unchanged and the digests must be **unmoved** — a moved digest
    /// there is a defect. `rotating` rotates which slot holds the newest token,
    /// which rotates the order `sum_keepdim` walks, so its digests **move
    /// deliberately** — the shape `--attn sdpa` has (§2.3.8c). §10.2g records
    /// the discriminator runs that separate that from a computational bug.
    #[arg(long, default_value = "shuffle")]
    conv_state: String,

    /// Fuse SiLU onto `w1`'s GEMV store (#484's kernel, wired by #509).
    ///
    /// **The digests MOVE across this axis, and that is the question this flag
    /// exists to answer.** The fused kernel applies SiLU to the f32
    /// accumulator where the `silu_f16` dispatch it replaces applies it at `T`
    /// to the already-rounded store — strictly fewer roundings and a different
    /// computation (§12.2d). So this has `--attn sdpa`'s shape rather than
    /// `--kv-append in-place`'s: a moved digest is the *expected* outcome and
    /// owes §2.3.5a's discriminator, and an **unmoved** one would mean the arm
    /// never engaged.
    #[arg(long, default_value_t = false)]
    gemv_silu: bool,

    /// Fuse the two residual adds onto their producing GEMV's store (#509).
    ///
    /// The producers are `out_proj`, `conv.out_proj` and `w2` — 60 dispatches
    /// per token (§12.2e). **Needs no compile flag since #516**; it used to
    /// require `-DLLOOM_509_RESIDUAL_EPILOGUE` through
    /// `LLOOM_434_METAL_CFLAGS` and failed at pipeline load without it.
    ///
    /// **The digests MOVE**, for `--gemv-silu`'s reason: the add lands at the
    /// f32 accumulator where `badd_f16` adds at `T` on the rounded store
    /// (§12.2h).
    #[arg(long, default_value_t = false)]
    gemv_residual: bool,

    /// Fold the decode conv's `broadcast_mul + sum_keepdim` into one depthwise
    /// `conv1d` dispatch (§12.2 row 1). **The digests MOVE** — the depthwise
    /// kernel accumulates serially ascending in f32 where `sum_keepdim` uses a
    /// `simd_sum` tree — so `--cpu --conv-fused` against `--cpu` is the §2.3.5a
    /// discriminator that says it is a reordering rather than a defect.
    #[arg(long, default_value_t = false)]
    conv_fused: bool,

    /// How the Q/K/V `transpose` views reach the norms and the KV append
    /// (`QkvLayout`, `DESIGN.md` §6.2c): `contiguous`, `strided` or `fused`. Every arm is
    /// meant to leave both digests where `contiguous` has them.
    #[arg(long, default_value = "contiguous")]
    qkv_layout: String,

    /// Carry the conv state across a turn-boundary prefill as the convolution's
    /// left context (`ConvPrefillCtx`, `DESIGN.md` §13.2f defect 1, #677). Off,
    /// every `--turns >= 2` pass zero-pads where the previous turn's last
    /// `l_cache - 1` tokens belong. Off by default: the canonical §2.3.8c pairs
    /// belong to the zero-padded arm (§7.1a). Moves both digests at `--turns 3`
    /// and neither at `--turns 1`.
    #[arg(long, default_value_t = false)]
    conv_prefill_ctx: bool,

    /// Who selects the GEMV epilogue fusions (`Fusion`, issue #732): `manual`
    /// (`--gemv-silu`/`--gemv-residual` as given) or `derived` (the optimizer
    /// derives the plan from the decode step's IR at construction and sets
    /// the two flags from it). A `derived` run is meant to reproduce the
    /// digests of the `manual` run whose flags match its plan.
    #[arg(long, default_value = "manual")]
    fusion: String,

    /// Which SiLU implementation the 30 `usilu` dispatches resolve to
    /// (issue #599, `DESIGN.md` §12.2l).
    ///
    /// `exact` | `recip` | `literal` | `coarse4` | `coarse1` | `hardswish`,
    /// in ascending error order. **`exact` is the default and resolves the
    /// same `silu_f16` name it always did**, so an unconfigured run is
    /// byte-identical (§7.1a).
    ///
    /// **Every arm but `exact` and `recip` MOVES THE DIGESTS by construction**
    /// — that is what an approximation to SiLU does, and it is the point.
    /// `recip` is the control: #604 measured it bit-identical, so an
    /// unmoved digest there is the arm engaging correctly and a MOVED one is
    /// a defect. Pair with `--force-tokens` and `--dump-logits` for §2.3.5a's
    /// magnitude ladder, which is the quantity #599's absorption curve wants.
    #[arg(long, default_value = "exact")]
    silu_impl: String,

    /// How every kernel's scalars reach it: `split` (the default, one `setBytes`
    /// per scalar) or `packed` (one `device const Params*`, issue #115).
    ///
    /// The families converted in #38 through #81 each carry both entry points
    /// from one body, and until now **nothing selected the packed ones** --
    /// `candle-core` calls only the classical `call_*`, which pass
    /// `ParamStyle::default()`. So the packed variants were compiled, checked by
    /// their own parity arms, and never dispatched by the model.
    ///
    /// This flag makes them reachable, and the digest is why it exists. Packing
    /// moves a scalar from an inline `setBytes` into a struct field at a
    /// computed offset, and §11.3d records that a field at the wrong offset does
    /// not crash -- the kernel reads a well-formed number from the wrong place
    /// and computes a plausible wrong answer. The per-family layout checks
    /// compare `sizeof` and every offset across the language boundary; what they
    /// cannot show is that the *whole decode path* still computes the same thing
    /// when every family switches at once. Only the §15.1 #7 gate does that.
    ///
    /// Expected result is therefore **bit-identical**, not merely stable. Both
    /// styles are instantiated from the same kernel body and differ only in how
    /// the arguments arrive, so unlike `--attn` this changes no arithmetic: a
    /// digest that moves here is a defect, not a variant (§2.3.5a).
    #[arg(long, default_value = "split")]
    param_style: String,

    /// Replay the stable, packed subset of a decode step from an
    /// `MTLIndirectCommandBuffer` (issue #115, `DESIGN.md` §17 Phase 2 item 10).
    ///
    /// **This flag is why the gate exists for this axis.** Replay suppresses a
    /// classical dispatch and runs a pre-encoded command in its place, and an
    /// ICB's commands are `ConcurrentDispatch` with no barrier between them
    /// (§3.5) -- so the ordering candle's `auto_barrier` emits has to be
    /// re-expressed on the commands themselves. A missing edge there is silent
    /// corruption rather than an error, and no barrier count can detect it: the
    /// count would simply be lower and still look plausible.
    ///
    /// Expected result is **bit-identical to the same configuration without
    /// `--icb`**. Replay changes where a dispatch is encoded, not what it
    /// computes, so a moved digest is a defect and not a variant (§2.3.5a).
    ///
    /// Needs `--param-style packed`: an ICB command cannot carry an inline
    /// constant (§3.7c), so under `split` every position is excluded and the run
    /// would gate an executor that replayed nothing.
    #[arg(long)]
    icb: bool,

    /// How far back an ICB command's barrier scan looks: `run-start` or
    /// `since-barrier`.
    ///
    /// `DESIGN.md` §11.3l's open question 3. `run-start` (the default, and what
    /// #115 shipped) emits a barrier wherever a command conflicts with anything
    /// earlier in its run. `since-barrier` scans back only to the previous
    /// barrier, which is what candle's `auto_barrier` does -- it emits one and
    /// then *replaces* `prev_outputs`, so an edge an earlier barrier already
    /// covers is not re-ordered.
    ///
    /// Both must produce the same digest: `setBarrier` orders a command after
    /// every command before it, so the transitive order is identical and only
    /// the number of commands carrying the flag differs. A moved digest here is
    /// a defect (§2.3.5a) and, per §3.5, the barrier count cannot detect one --
    /// which is why this is gated on the digest and not on the count.
    #[arg(long, default_value = "run-start")]
    icb_barrier_scope: String,

    /// Whether candle's own `auto_barrier` still fires at a replayed position:
    /// `always` or `skip-replayed` (issue #144).
    ///
    /// `DESIGN.md` §11.3p. `auto_barrier` is the first statement of
    /// `dispatch_thread_groups` and `offer_to_executor` is consulted after it,
    /// so a replayed position emits candle's barrier *and* the ICB's -- §11.3n
    /// measures the 401 as **additive**, for 906 orderings against the classical
    /// arm's 505. §11.3p attributes the 505 and finds **393 fire at covered
    /// non-head positions**, where the ICB has already encoded ordering.
    ///
    /// `always` is the default and is byte-for-byte what ships.
    /// `skip-replayed` suppresses candle's barrier at a **non-head member of a
    /// run already in flight** -- narrower than "the position is covered", for
    /// the three reasons §11.3p gives and which
    /// `measurements/issue-144-predicate.md` argues at edge level.
    ///
    /// **This removes ordering edges**, so under `HazardTrackingModeUntracked` a
    /// wrongly-removed one is silent corruption rather than an error (§3.5) --
    /// the same standing as `HazardKey::Range`. Both arms must produce the
    /// canonical digests; a moved digest here is a defect and not a variant
    /// (§2.3.5a), and the barrier count cannot detect one because it would
    /// simply be lower and still plausible (§2.4).
    #[arg(long, default_value = "always")]
    icb_replay_barriers: String,
}

/// Follow-up prompts for multi-turn mode, cycled in order.
///
/// Fixed and content-free with respect to the measurement: they exist to keep
/// the model generating real text across a long conversation, not to test any
/// particular capability.
const FOLLOW_UPS: [&str; 8] = [
    "Expand on that, and give a concrete example.",
    "What are the main tradeoffs involved?",
    "How did this evolve historically?",
    "What do people most often get wrong about it?",
    "Compare that with the main alternative approach.",
    "What would you measure to tell if it is working well?",
    "Summarize the key points so far.",
    "What is the next thing someone should learn about this?",
];

/// Parse a `--force-tokens` file into the sequence to decode.
///
/// Accepts one id per line and a whitespace- or comma-separated list on one
/// line, so this harness's own `TOKENS` output round-trips without editing.
/// `#` starts a comment.
///
/// Parse errors are refused rather than skipped: a file with one unparseable
/// entry silently dropped would force a *different, shorter* sequence than the
/// one on disk, and both arms would agree on it — a comparison that is
/// internally consistent and not the one that was asked for (§2.4).
/// Write one step's raw logits to `--dump-logits`, in the format the ordinary
/// decode path writes.
///
/// Factored out when the speculative arm (issue #89) became a second producer of
/// steps: two copies of this would be two chances for the two arms' dumps to
/// disagree in *format*, which `tools/logit-delta/` would then read as a
/// difference in *values*. One writer, one format.
fn dump_step(
    sink: &mut Option<std::io::BufWriter<std::fs::File>>,
    step: usize,
    values: &[f32],
) -> Result<()> {
    let Some(sink) = sink.as_mut() else {
        return Ok(());
    };
    use std::io::Write;
    let mut line = String::with_capacity(values.len() * 12);
    line.push_str(&format!("step {step}"));
    for v in values {
        // Hex of the bit pattern: exact, so the comparison is over the values
        // the model produced rather than over a decimal rendering of them.
        line.push_str(&format!(" {:08x}", v.to_bits()));
    }
    line.push('\n');
    sink.write_all(line.as_bytes())
        .context("writing --dump-logits")?;
    Ok(())
}

fn parse_forced_tokens(raw: &str) -> Result<Vec<u32>> {
    let mut out = Vec::new();
    for (lineno, line) in raw.lines().enumerate() {
        let line = match line.split_once('#') {
            Some((before, _)) => before,
            None => line,
        };
        for field in line.split([',', ' ', '\t']).filter(|f| !f.is_empty()) {
            let id: u32 = field
                .parse()
                .with_context(|| format!("line {}: {field:?} is not a token id", lineno + 1))?;
            out.push(id);
        }
    }
    anyhow::ensure!(!out.is_empty(), "the token file parsed to no tokens");
    Ok(out)
}

fn main() -> Result<()> {
    let args = Args::parse();

    if args.self_test {
        sha256::self_test().map_err(|e| anyhow::anyhow!("sha256 self-test failed: {e}"))?;
        println!("sha256 self-test: OK (NIST vectors + 1e6*'a' + streaming)");
        return Ok(());
    }

    let model_dir = args.model_dir.or_else(default_model_dir).context(
        "no --model-dir given and the LFM2.5-VL-3B snapshot was not found in the HF cache",
    )?;

    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::new_metal(0)?
    };

    // f16 on Metal is what ambrogio loads. Measuring f32 here would measure a
    // path nothing runs.
    //
    // This comment used to add: "LFM2 ships bf16, but Metal's bf16 kernel
    // coverage is patchy and unsupported ops fall back to the CPU silently."
    // **That is false, and it had never been tested** -- it was quoted into
    // `DESIGN.md` §9.1b as a fact and made load-bearing for three decisions.
    // Measured (#307, §3.9): 12 of 12 decode kernel families dispatch a native
    // bf16 sibling at an IDENTICAL count (1135 against 1135), `lfm2-smoke`
    // PASSes at bf16 on all three `--attn` arms with byte-identical text, and
    // **no dispatch falls back to the CPU on any arm**. `--dtype bf16` works.
    //
    // The default stays f16 on the honest reason, which is the one this comment
    // led with: it is what the consumer loads, so it is the path worth gating.
    // Not on a claim about coverage -- and a default is flipped by its own
    // argued decision (§7.1a), which this survey is not.
    //
    // One real gap, and it is hardware rather than a missing instantiation:
    // `bfloat` has NO simdgroup operations at all -- 12 of 12 absent where
    // `half` has them -- so every bf16 reduction takes a 5-instruction shuffle
    // tree where f16 takes one `simd_sum`. Emulated, fixed-order, correct, and
    // unpriced in time (§3.9).
    let dtype = match args.dtype.as_deref() {
        Some("f16") => DType::F16,
        Some("f32") => DType::F32,
        Some("bf16") => DType::BF16,
        Some(other) => anyhow::bail!("unknown dtype {other}"),
        None => match &device {
            Device::Metal(_) => DType::F16,
            Device::Cuda(_) => DType::BF16,
            Device::Cpu => DType::F32,
        },
    };

    let config_raw = std::fs::read_to_string(model_dir.join("config.json"))
        .with_context(|| format!("reading {}", model_dir.join("config.json").display()))?;
    let mut config = parse_config(&config_raw)?;

    // Refused rather than defaulted. A silent fallback would report the generic
    // path's digests under the sdpa arm's name, and the whole point of this run
    // is to tell those two apart -- #69's vacuous determinism run is exactly the
    // failure this avoids (§2.4).
    config.attn_impl = match args.attn.as_str() {
        "generic" => AttnImpl::Generic,
        "sdpa" => AttnImpl::Sdpa,
        // Issue #116. Selectable and never a default: 10.4's argument for it
        // is structural rather than measured, and the kv_len at which it pays
        // is #61's to find.
        "flash" => AttnImpl::FlashDecoding,
        other => anyhow::bail!("--attn must be `generic`, `sdpa` or `flash`, got `{other}`"),
    };
    config.flash_page_size = args.flash_page_size;
    config.flash_pages_per_chunk = args.flash_k;
    // §9.1a's sizing policy (#234). Refused rather than defaulted, for the same
    // reason as `--attn`: a silently-ignored flag makes a gate report on an arm
    // it did not run (§2.4).
    config.flash_scratch_sizing =
        lloom::nn::ops::FlashScratchSizing::parse(&args.flash_scratch_sizing)
            .map_err(anyhow::Error::msg)?;
    println!("attention implementation: {:?}", config.attn_impl);
    // Refused rather than defaulted, for the same reason as `--attn`: an
    // unrecognized value that silently pinned the default would report the
    // `cat` arm's digest under the `in-place` arm's name, which is #69's
    // vacuous run (§2.4, §9.2f).
    config.kv_append = match args.kv_append.as_str() {
        "paged" => lloom::models::lfm2::KvAppend::Paged,
        "cat" => KvAppend::Cat,
        "in-place" => KvAppend::InPlace,
        other => anyhow::bail!("--kv-append must be `cat` or `in-place`, got `{other}`"),
    };
    println!("kv append: {:?}", config.kv_append);

    // Refused rather than defaulted, for `--kv-append`'s reason: an
    // unrecognized value that silently pinned the default would report the f16
    // arm's digest under a quantized arm's name, which is #69's vacuous run.
    config.kv_quant = KvQuant::parse(&args.kv_quant).map_err(anyhow::Error::msg)?;
    println!("kv quant: {:?}", config.kv_quant);

    config.conv_state = ConvState::parse(&args.conv_state).map_err(anyhow::Error::msg)?;
    println!("conv state: {:?}", config.conv_state);

    // #599's SiLU arm. Refused rather than defaulted, for `--conv-state`'s
    // reason: an unrecognized value silently pinning `exact` would report the
    // shipping arm's digest under a variant's name, which is #69's vacuous run
    // (§2.4). Printed whether or not it is `exact` — a digest is only
    // interpretable against the arm it was taken in.
    let silu = candle::kernels::SiluImpl::parse(&args.silu_impl).ok_or_else(|| {
        anyhow::anyhow!(
            "--silu-impl must be one of `exact`, `recip`, `literal`, `coarse4`, \
             `coarse1`, `hardswish`, got `{}`",
            args.silu_impl
        )
    })?;
    // The SiLU arm now rides `Config::silu_impl` (RESTRUCTURING §1.6): the field
    // is the source of truth and `Model::new` pushes it into the kernel cache,
    // so setting it here (rather than calling `set_silu_impl` directly) is what
    // makes the run record it. Setting the atomic directly would be clobbered by
    // `Model::new`'s push of `config.silu_impl`.
    config.silu_impl = silu;
    println!("silu impl: {}", silu.name());

    // The two GEMV epilogue fusions (#509's wiring of #484's and #509's
    // kernels). Printed whether or not they are set, because *inert* and
    // *unrecorded* are different facts (#241) and a digest is only
    // interpretable against the arm it was taken in (§2.3.7).
    config.gemv_silu = args.gemv_silu;
    config.gemv_residual = args.gemv_residual;
    config.conv_fused = args.conv_fused;
    config.conv_prefill_ctx = args.conv_prefill_ctx;
    config.qkv_layout =
        lloom::models::lfm2::QkvLayout::parse(&args.qkv_layout).map_err(anyhow::Error::msg)?;
    println!("qkv layout: {:?}", config.qkv_layout);
    config.fusion =
        lloom::models::lfm2::FusionPolicy::parse(&args.fusion).map_err(anyhow::Error::msg)?;
    println!(
        "gemv epilogue: silu={} residual={} fusion={}",
        config.gemv_silu,
        config.gemv_residual,
        config.fusion.name()
    );

    // The binding-style axis (issue #115). Set before any dispatch, and read
    // back from the crate rather than echoed from `args`, so the line printed is
    // what the kernels will actually use: an A/B behind a switch owes a check
    // that the two arms differ in something observable (§2.4, §9.2f).
    #[cfg(feature = "metal")]
    {
        use candle::kernels::{default_param_style, set_default_param_style, ParamStyle};
        let want = match args.param_style.as_str() {
            "split" => ParamStyle::Split,
            "packed" => ParamStyle::Packed,
            other => anyhow::bail!("--param-style must be `split` or `packed`, got `{other}`"),
        };
        set_default_param_style(want);
        let got = default_param_style();
        // Not `assert_eq!` against `want` alone: that would pass if the setter
        // and the getter agreed with each other while neither reached the
        // kernels. The load-bearing evidence that this arm engaged is the
        // kernel-name census `lfm2-dispatch-trace` prints -- every packed
        // dispatch resolves a `*_packed` `[[host_name]]` -- and this is the
        // cheap precondition for it.
        anyhow::ensure!(
            got == want,
            "--param-style {want:?} did not take effect; default_param_style() reports {got:?}"
        );
        println!("param style: {got:?}");
    }

    // ICB replay (issue #115). Both switches have to be thrown before the first
    // pipeline is built: `supportIndirectCommandBuffers` is a property of a
    // pipeline and pipelines are cached per process, so one built earlier keeps
    // the old setting -- and encoding that into an ICB is §3.7d's segfault at
    // encode time, with no error to catch.
    #[cfg(feature = "metal")]
    if args.icb {
        anyhow::ensure!(
            args.param_style == "packed",
            "--icb needs --param-style packed: an ICB command cannot carry an inline constant \
             (DESIGN.md §3.7c), so under `split` every position is excluded and this would \
             gate an executor that replayed nothing"
        );
        candle::kernels::set_pipelines_support_icb(true)
            .map_err(|e| anyhow::anyhow!("enabling supportIndirectCommandBuffers: {e}"))?;
        candle::kernels::set_constants_pool_enabled(true);
        println!("icb: enabled");
    }
    #[cfg(not(feature = "metal"))]
    anyhow::ensure!(!args.icb, "--icb needs the `metal` feature");
    // Refused rather than ignored off Metal: silently accepting `--param-style
    // packed` on a CPU build and reporting a passing digest for the default path
    // is #69's vacuous run exactly (§2.4).
    #[cfg(not(feature = "metal"))]
    anyhow::ensure!(
        args.param_style == "split",
        "--param-style {:?} needs the `metal` feature; this binary has no packed entry points",
        args.param_style
    );

    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("loading tokenizer: {e}"))?;

    let files = weight_files(&model_dir)?;
    let names = tensor_names(&files[0])?;
    let nested = names.iter().any(|n| n.starts_with("model.language_model."))
        && !names.iter().any(|n| n == "model.embed_tokens.weight");

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, &device)? };
    // A VL checkpoint nests the language stack under `model.language_model.`;
    // candle's text-only loader asks for `model.*`.
    let vb = if nested {
        vb.rename_f(|name: &str| match name.strip_prefix("model.") {
            Some(rest) => format!("model.language_model.{rest}"),
            None => name.to_string(),
        })
    } else {
        vb
    };

    let model = Model::new(&config, vb).context("constructing LFM2 model")?;
    let mut cache = Cache::new(true, dtype, &config, &device).context("allocating KV cache")?;

    let sampling = if args.temperature <= 0. {
        Sampling::ArgMax
    } else {
        Sampling::TopP {
            p: args.top_p,
            temperature: args.temperature,
        }
    };
    let mut logits_processor = LogitsProcessor::from_sampling(args.seed, sampling);

    let mut eos_ids: Vec<u32> = config.eos_token_id.into_iter().collect();
    for tok in ["<|im_end|>", "<|endoftext|>"] {
        if let Some(id) = tokenizer.token_to_id(tok) {
            eos_ids.push(id);
        }
    }

    // The checkpoint's chat_template.jinja reduces, for a single user turn with
    // no system prompt and no tools, to exactly this. Constructing it directly
    // avoids a Jinja dependency while still feeding the model the framing it
    // was trained on — fed a bare string it emits EOS at step 0.
    let prompt = if args.raw_prompt {
        args.prompt.clone()
    } else {
        format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            args.prompt
        )
    };

    let prompt_ids = tokenizer
        .encode(prompt.as_str(), true)
        .map_err(|e| anyhow::anyhow!("tokenizing prompt: {e}"))?
        .get_ids()
        .to_vec();
    anyhow::ensure!(!prompt_ids.is_empty(), "prompt tokenized to nothing");

    // Issue #170. Loaded before the loop so a missing or malformed file fails
    // before a forward pass rather than after one.
    let forced: Option<Vec<u32>> = match args.force_tokens.as_deref() {
        Some(p) => {
            let raw = std::fs::read_to_string(p)
                .with_context(|| format!("reading --force-tokens {p}"))?;
            let ids = parse_forced_tokens(&raw).with_context(|| format!("parsing {p}"))?;
            println!("force tokens: {} ids from {p}", ids.len());
            Some(ids)
        }
        None => None,
    };
    // How many steps took the forced id, and at how many of those the sampler
    // would have chosen something else.
    //
    // The first is engagement proof and the second is the measurement. §2.4
    // requires an instrument be shown to have engaged from a quantity it should
    // have moved — #69's vacuous determinism run consumed the `OnceLock`
    // guarding its env switch, so both arms ran the default and the "changed"
    // arm reported a passing digest for the unchanged path. A `forced=0` run
    // that printed a digest would be that failure exactly: the sampled path's
    // result wearing the forced arm's name.
    let mut forced_steps = 0usize;
    let mut forced_overrides = 0usize;

    // Issue #89. `windows` and `proposed`/`accepted` are the acceptance rate,
    // which is what a cost model needs beside a timing; `corrupted` is the
    // wrong-proposer arm's engagement proof, in the shape §2.4 requires — an arm
    // named `wrong` that corrupted nothing would be the oracle arm wearing the
    // rollback arm's name, which is #69's vacuous run in a new quantity.
    let mut spec_windows = 0usize;
    let mut spec_proposed = 0usize;
    let mut spec_accepted = 0usize;
    let mut spec_corrupted = 0usize;
    let spec_wrong_every: Option<usize> = match args.spec_proposer.as_str() {
        "oracle" => None,
        s => match s.strip_prefix("wrong:") {
            Some(n) => {
                let n: usize = n.parse().map_err(|_| {
                    anyhow::anyhow!("--spec-proposer wrong:<N> needs an integer, got {s:?}")
                })?;
                if n == 0 {
                    anyhow::bail!("--spec-proposer wrong:0 would corrupt nothing");
                }
                Some(n)
            }
            None => anyhow::bail!(
                "--spec-proposer must be `oracle` or `wrong:<N>`, got {:?}",
                args.spec_proposer
            ),
        },
    };
    let vocab_size = config.vocab_size as u32;

    let mut token_hasher = sha256::Sha256::new();
    let mut logits_hasher = sha256::Sha256::new();
    let mut tokens: Vec<u32> = Vec::with_capacity(args.n);
    let mut per_step: Vec<(u32, String, usize)> = Vec::with_capacity(args.n);
    let mut logits_sink = match args.dump_logits.as_deref() {
        Some(p) => Some(std::io::BufWriter::new(
            std::fs::File::create(p).with_context(|| format!("creating {p}"))?,
        )),
        None => None,
    };

    let start = std::time::Instant::now();

    // kv_len at the moment each token is produced. Reported alongside the
    // divergence position, because a drift that localizes to a page boundary or
    // a chunk-count change is the debugging signal (§2.3.6).
    let mut kv_len = 0usize;
    let mut hit_eos = false;
    let mut turns_run = 0usize;

    // Arena state (`DESIGN.md` §9.2; issues #69, #70).
    let metal_device = match &device {
        Device::Metal(d) => Some(d.clone()),
        _ => None,
    };
    let arena_layout = match args.arena_layout.as_str() {
        "packed" => ArenaLayout::Packed,
        "non-aliasing" | "nonaliasing" | "reference" => ArenaLayout::NonAliasing,
        other => anyhow::bail!("unknown --arena-layout {other:?}; want packed or non-aliasing"),
    };
    let gpu_offsets = match args.arena_offsets.as_str() {
        "cpu" => false,
        "gpu" => true,
        other => anyhow::bail!("unknown --arena-offsets {other:?}; want cpu or gpu"),
    };
    if gpu_offsets {
        if !args.arena {
            anyhow::bail!("--arena-offsets gpu needs --arena");
        }
        if arena_layout != ArenaLayout::NonAliasing {
            // Refused rather than downgraded: a packed plan reuses slots, so its
            // offsets are not monotone and no forward-only cursor can reproduce
            // them. Falling back silently would report the CPU path's digest
            // under the GPU path's name, which is the failure §2.4 records.
            anyhow::bail!(
                "--arena-offsets gpu needs --arena-layout non-aliasing: a packed plan reuses \
                 slots, so its offsets are not monotone"
            );
        }
    }
    /// Decode steps recorded before the arena is installed. Two, because
    /// comparing two is what separates an activation from session state.
    const ARENA_RECORD_STEPS: usize = 2;

    // Steps the ICB executor records before deciding what is replayable. Three
    // rather than the minimum of two: a position that happens to agree between
    // two consecutive steps but drifts on the third would be admitted by a
    // two-step window, and admitting one wrongly is silent corruption (§3.5)
    // rather than a visible failure.
    #[cfg(feature = "metal")]
    const ICB_RECORD_STEPS: usize = 3;
    #[cfg(feature = "metal")]
    let icb_barrier_scope = match args.icb_barrier_scope.as_str() {
        "run-start" => candle::kernels::BarrierScope::RunStart,
        "since-barrier" => candle::kernels::BarrierScope::SinceBarrier,
        other => anyhow::bail!(
            "--icb-barrier-scope must be `run-start` or `since-barrier`, got `{other}`"
        ),
    };
    #[cfg(feature = "metal")]
    let icb_replay_barriers = match args.icb_replay_barriers.as_str() {
        "always" => candle::kernels::ReplayBarriers::Always,
        "skip-replayed" => candle::kernels::ReplayBarriers::SkipReplayed,
        other => anyhow::bail!(
            "--icb-replay-barriers must be `always` or `skip-replayed`, got `{other}`"
        ),
    };
    #[cfg(feature = "metal")]
    let icb_executor = args.icb.then(|| {
        candle::kernels::IcbExecutor::configured(
            ICB_RECORD_STEPS,
            icb_barrier_scope,
            icb_replay_barriers,
        )
    });
    #[cfg(feature = "metal")]
    let icb_installed_at = if args.arena { ARENA_RECORD_STEPS } else { 0 };

    let mut decode_steps = 0usize;

    'turns: for turn in 0..args.turns.max(1) {
        // Turn 0 prefills the initial prompt; later turns append a follow-up to
        // the same cache, so KV state is reused rather than rebuilt.
        let turn_ids: Vec<u32> = if turn == 0 {
            prompt_ids.clone()
        } else {
            let follow_up = FOLLOW_UPS[(turn - 1) % FOLLOW_UPS.len()];
            let text = format!("<|im_start|>user\n{follow_up}<|im_end|>\n<|im_start|>assistant\n");
            tokenizer
                .encode(text.as_str(), false)
                .map_err(|e| anyhow::anyhow!("tokenizing follow-up: {e}"))?
                .get_ids()
                .to_vec()
        };

        let input = Tensor::new(turn_ids.as_slice(), &device)?.unsqueeze(0)?;
        let mut logits = model
            .forward(&input, kv_len, &mut cache)
            .context("prefill forward pass")?
            .squeeze(0)?;
        kv_len += turn_ids.len();
        turns_run = turn + 1;

        // ---- the speculative arm (issue #89) -------------------------------
        //
        // A separate loop rather than a branch inside the one below, because the
        // shapes genuinely differ: an ordinary step consumes one token and emits
        // one, where a verify pass consumes K and emits between 1 and K. Folding
        // them would put a `seq_len`-dependent branch on the path every recorded
        // digest belongs to, to serve a caller that path does not have — which is
        // §11.3l finding 4's shape.
        if let Some(k) = args.speculate {
            let Some(oracle) = forced.as_ref() else {
                anyhow::bail!(
                    "--speculate needs --force-tokens: the proposer is a known-correct \
                     sequence, which is what lets the mechanism be measured without a \
                     draft model (issue #89)"
                );
            };
            // **A position predicts the token that FOLLOWS the one it reads**,
            // and getting the resulting off-by-one wrong is the failure this
            // loop is shaped to avoid. Spelled out, because it is the whole of
            // why a `k`-position pass yields `k` tokens rather than `k - 1`:
            //
            // * the prediction for the *next* step is already in hand — it is
            //   `logits`, carried from the prefill or from the previous pass's
            //   last position. That token costs nothing and is emitted first.
            //   Feeding it back is what conditions the pass on it.
            // * the pass then feeds `[emitted, prop_1, .., prop_{w-1}]`, so
            //   position `j` reads `ids[j]` and predicts step `d + j + 1`. It
            //   verifies `w - 1` proposals and predicts one more — the bonus.
            //
            // A first draft compared position `j`'s argmax against `window[j]`
            // rather than against the proposal for the step it predicts. The
            // symptom was not a crash or a wrong shape: it produced a full,
            // stable token stream with `accept_rate = 0.26` under a **perfect**
            // oracle, which reads as a bad proposer rather than as a
            // misalignment. That is why the oracle arm is the first gate and why
            // its acceptance rate is asserted rather than merely printed —
            // §2.4's rule that an instrument must be shown to have engaged, in
            // the one quantity that can distinguish these two.
            loop {
                let done = per_step.len();
                if tokens.len() >= args.n || done >= oracle.len() {
                    break 'turns;
                }

                // Step `done`'s token, from the logits already in hand.
                let carried = logits.to_dtype(DType::F32)?.to_vec1::<f32>()?;
                let mut carried_bytes = Vec::with_capacity(carried.len() * 4);
                for v in &carried {
                    carried_bytes.extend_from_slice(&v.to_bits().to_le_bytes());
                }
                let carried_digest = sha256::digest_hex(&carried_bytes);
                dump_step(&mut logits_sink, per_step.len(), &carried)?;
                let emitted = logits_processor.sample(&logits).context("sampling")?;
                forced_steps += 1;
                if emitted != oracle[done] {
                    forced_overrides += 1;
                }
                logits_hasher.update(&carried_bytes);
                token_hasher.update(&emitted.to_le_bytes());
                tokens.push(emitted);
                per_step.push((emitted, carried_digest, kv_len));

                if tokens.len() >= args.n || per_step.len() >= oracle.len() {
                    break 'turns;
                }

                // **The proposals are indexed by the step they propose, and the
                // oracle is indexed the same way — which is only sound because
                // the emitted stream and the oracle agree.**
                //
                // Under greedy that identity is the mechanism's own guarantee:
                // every emitted token is the target's argmax, which is what
                // non-speculative decoding emits, which is what the oracle
                // recorded. A rejection changes *which proposals are wasted*,
                // never *what is emitted*. So a divergence here is not a wrong
                // proposal — it is the mechanism failing, and the assertion says
                // so rather than letting the run continue against proposals for
                // positions the model never reached.
                anyhow::ensure!(
                    emitted == oracle[done],
                    "speculative decoding diverged from the non-speculative sequence at \
                     step {done}: emitted {emitted}, reference {}. Under greedy decoding \
                     these are identical by construction (issue #89), so this is a defect \
                     in the verifier and not a rejected proposal.",
                    oracle[done]
                );

                // The proposals for the steps after it, clipped to the oracle.
                let want = k.min(oracle.len() - per_step.len());
                let mut props: Vec<u32> = oracle[done + 1..done + 1 + want].to_vec();
                if let Some(every) = spec_wrong_every {
                    // The deliberately-wrong proposer: corrupt every `N`-th
                    // proposal, so a rejection happens on a schedule rather than
                    // by chance and the rollback is exercised on most windows.
                    for (j, p) in props.iter_mut().enumerate() {
                        if (done + 1 + j) % every == every - 1 {
                            *p = p.wrapping_add(1) % vocab_size;
                            spec_corrupted += 1;
                        }
                    }
                }

                // `[emitted, props[..want-1]]` — `want` positions, verifying
                // `want - 1` proposals and predicting one bonus token.
                let width = props.len();
                let mut ids = Vec::with_capacity(width);
                ids.push(emitted);
                ids.extend_from_slice(&props[..width - 1]);

                let tok = cache.advance(width)?;
                let input = Tensor::new(ids.as_slice(), &device)?.unsqueeze(0)?;
                let all = model
                    .forward_all(&input, kv_len, &mut cache)
                    .context("speculative verify pass")?
                    .squeeze(0)?;
                kv_len += width;

                // Find the accepted prefix first, *then* emit it.
                //
                // **The last accepted position's prediction is carried, not
                // emitted**, and separating the two loops is what makes that
                // hard to get wrong. A single loop that both emitted position
                // `j` and assigned `logits = row` emitted the same prediction
                // twice — once from the pass and once from the next iteration's
                // carried-emit — which showed up as two consecutive steps with
                // an *identical logits digest* and an acceptance rate that
                // looked like a mediocre proposer. Two loops make the invariant
                // structural: exactly one row is carried and every other
                // accepted row is emitted.
                let mut accepted = 0usize; // positions whose state survives
                for j in 0..width {
                    accepted += 1;
                    let target = logits_processor.sample(&all.i(j)?).context("sampling")?;
                    if target != props[j] {
                        break;
                    }
                }
                // Emit every accepted position except the last, whose logits
                // become the carried prediction for the next window's first
                // token. Clipped so a run stops mid-window at `--n`.
                let mut carry = None;
                for j in 0..accepted {
                    let row = all.i(j)?;
                    if j + 1 == accepted {
                        carry = Some(row);
                        break;
                    }
                    if tokens.len() >= args.n || per_step.len() >= oracle.len() {
                        break;
                    }
                    let step_logits = row.to_dtype(DType::F32)?.to_vec1::<f32>()?;
                    let mut step_bytes = Vec::with_capacity(step_logits.len() * 4);
                    for v in &step_logits {
                        step_bytes.extend_from_slice(&v.to_bits().to_le_bytes());
                    }
                    let step_digest = sha256::digest_hex(&step_bytes);
                    dump_step(&mut logits_sink, per_step.len(), &step_logits)?;
                    let target = logits_processor.sample(&row).context("sampling")?;
                    forced_steps += 1;
                    if target != oracle[per_step.len()] {
                        forced_overrides += 1;
                    }
                    logits_hasher.update(&step_bytes);
                    token_hasher.update(&target.to_le_bytes());
                    tokens.push(target);
                    per_step.push((target, step_digest, kv_len));
                }
                if let Some(row) = carry {
                    logits = row;
                }

                // The pass appended `width` positions and `accepted` of them
                // produced tokens that were kept, so `width - accepted` are
                // discarded — a length decrement and a phase move, neither of
                // which clears a byte (`DESIGN.md` §10.2a).
                cache.resolve(tok, accepted)?;
                kv_len -= width - accepted;
                spec_windows += 1;
                spec_accepted += accepted;
                spec_proposed += width;
                decode_steps += 1;
            }
        }

        loop {
            if tokens.len() >= args.n {
                break 'turns;
            }
            // Stop where the forced sequence stops, checked before the pass
            // rather than after it. Continuing on the sampled token past the end
            // would leave one dump whose head is comparable and whose tail is
            // not, with nothing in the file marking where that changed.
            if let Some(ids) = forced.as_ref() {
                if per_step.len() >= ids.len() {
                    break 'turns;
                }
            }

            // Digest the logits before sampling: this is the quantity the
            // hardware actually produced, whereas the token is that quantity
            // after a threshold has thrown most of the information away.
            let step_logits = logits.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            let mut step_bytes = Vec::with_capacity(step_logits.len() * 4);
            for v in &step_logits {
                step_bytes.extend_from_slice(&v.to_bits().to_le_bytes());
            }
            let step_digest = sha256::digest_hex(&step_bytes);
            logits_hasher.update(&step_bytes);

            dump_step(&mut logits_sink, per_step.len(), &step_logits)?;

            // The sampler runs either way. Under `--force-tokens` its choice is
            // recorded and then discarded, which is what makes `overrides` a
            // measurement rather than a diagnostic: it counts the positions at
            // which this arm would have left the reference path.
            let sampled = logits_processor.sample(&logits).context("sampling")?;

            let next = match forced.as_ref().and_then(|ids| ids.get(per_step.len())) {
                Some(&id) => {
                    forced_steps += 1;
                    if id != sampled {
                        forced_overrides += 1;
                    }
                    id
                }
                // Not forcing. Unreachable *while* forcing: the loop head
                // already broke out at the end of the sequence.
                None => sampled,
            };

            // EOS and turn boundaries are suppressed under forcing, and the
            // suppression is the point rather than a convenience.
            //
            // Both are decisions taken *from the sampled token*, so leaving them
            // live would let the arms end a turn at different steps — after
            // which position `i` no longer names the same thing in the two dumps
            // and the comparison this flag exists to enable is invalid again,
            // silently. Forcing means the sequence is the sequence: the run ends
            // when the forced ids run out or `--n` is reached.
            if forced.is_none() && eos_ids.contains(&next) && !args.ignore_eos {
                // End of this turn. With more turns to run the conversation
                // continues; otherwise generation is done.
                hit_eos = true;
                if turn + 1 < args.turns.max(1) {
                    break;
                }
                break 'turns;
            }

            token_hasher.update(&next.to_le_bytes());
            tokens.push(next);
            per_step.push((next, step_digest, kv_len));

            let input = Tensor::new(&[next], &device)?.unsqueeze(0)?;

            // Arena bookkeeping, if one was asked for (`DESIGN.md` §9.2, issues
            // #69 and #70). The first two decode steps are recorded rather than
            // served: two are the minimum that can tell an activation from
            // session state, since what separates them is a size that moved
            // between steps (§9.1, #68 finding 4).
            if let (true, Some(dev)) = (args.arena, metal_device.as_ref()) {
                match decode_steps {
                    0 => dev.begin_arena_recording(),
                    n if n < ARENA_RECORD_STEPS => dev.next_arena_recording_step(),
                    _ => {}
                }
                dev.begin_decode_step();
            }

            // Install the ICB executor once the operands are stable. Under
            // `--arena` that is after its recording steps: recording earlier
            // would capture the pool's per-step buffer identities and exclude
            // every position as varying, for a reason that is not the
            // executor's.
            #[cfg(feature = "metal")]
            if let (Some(exec), Some(dev)) = (icb_executor.as_ref(), metal_device.as_ref()) {
                if decode_steps == icb_installed_at {
                    dev.set_executor(std::sync::Arc::new(
                        candle::kernels::metal::ExecutorSlot::Custom(exec.clone()),
                    ));
                }
            }

            logits = model
                .forward(&input, kv_len, &mut cache)
                .context("decode forward pass")?
                .squeeze(0)?;

            // Close the ICB step here: after the forward pass and before the
            // next iteration samples, so a step is exactly the forward pass.
            // Sampling adds a `softmax_f32` and an `affine_f32`, and including
            // them in some windows and not others makes position N a different
            // dispatch in different steps -- which reads as "everything varies"
            // rather than as a misaligned boundary.
            #[cfg(feature = "metal")]
            if let (Some(_), Some(dev)) = (icb_executor.as_ref(), metal_device.as_ref()) {
                if decode_steps >= icb_installed_at {
                    dev.end_executor_step()
                        .map_err(|e| anyhow::anyhow!("closing ICB step {decode_steps}: {e}"))?;
                }
            }

            if let (true, Some(dev)) = (args.arena, metal_device.as_ref()) {
                dev.end_decode_step();
                if decode_steps == ARENA_RECORD_STEPS - 1 {
                    // Read BEFORE `finish_arena_recording`, which `take`s the
                    // recorder the reservations are derived from (#47).
                    let reservations = if args.session_region {
                        dev.arena_recording_session_reservations(
                            kv_len,
                            lloom::models::lfm2::DEFAULT_KV_CAPACITY,
                        )
                    } else {
                        None
                    };
                    let plan = dev
                        .finish_arena_recording(arena_layout)
                        .context("arena recording produced no allocations")?;
                    let monotone = plan.is_bump_reproducible();
                    let (covered, total) = plan.covered();
                    dev.install_arena(plan, arena_layout)
                        .context("installing the arena")?;
                    eprintln!("arena: {covered} of {total} ordinals served ({arena_layout:?})");
                    // #47, after the arena because it serves what the arena
                    // declines: the offer order is arena, region, pool.
                    if let Some(res) = reservations {
                        let bytes = dev
                            .install_session_region(&res)
                            .context("installing the session region")?;
                        eprintln!("session region: {} tenants, {} B", res.len(), bytes);
                    }

                    if gpu_offsets {
                        if !monotone {
                            anyhow::bail!(
                                "the recorded plan's offsets are not monotone, so a bump \
                                 allocator cannot reproduce them"
                            );
                        }
                        let served = dev
                            .install_gpu_arena_offsets()
                            .context("computing arena offsets on the GPU")?;
                        // The engagement check §2.4 requires, and #69's vacuous
                        // determinism run is why it is here: a digest reported
                        // under a flag that silently did nothing is a passing
                        // result for the path the flag was meant to replace.
                        // `arena_offsets()` reads `Gpu` only after the table was
                        // verified equal to the plan element-wise.
                        eprintln!(
                            "arena: offsets computed on the GPU and verified ({served} ordinals); \
                             source now {:?}",
                            dev.arena_offsets()
                        );
                    }
                }
                decode_steps += 1;
            }

            kv_len += 1;
        }
    }

    let elapsed = start.elapsed();
    let token_digest = sha256::hex(&token_hasher.finalize());
    let logits_digest = sha256::hex(&logits_hasher.finalize());

    if args.dump_tokens {
        for (i, (tok, digest, kv)) in per_step.iter().enumerate() {
            println!("step {i} token {tok} kv_len {kv} logits {digest}");
        }
    }

    // Issue #170. Engagement proof for `--force-tokens`, in the shape §2.4
    // requires: a quantity the flag should have moved, checked rather than the
    // flag trusted. A run that forced nothing and printed a digest would be
    // reporting the sampled path's result under the forced arm's name, which is
    // #69's vacuous determinism run.
    if let Some(ids) = forced.as_ref() {
        println!(
            "FORCED requested={} forced={} overrides={} generated={}",
            ids.len(),
            forced_steps,
            forced_overrides,
            tokens.len(),
        );
        anyhow::ensure!(
            forced_steps > 0,
            "--force-tokens was passed and no step took a forced id, so the digest below is the \
             sampled path's wearing this arm's name"
        );
        // Not an error. `overrides == 0` means this arm sampled the reference
        // sequence unaided — which is the §10.2g case, where the arms did not
        // diverge in tokens and forcing was therefore not load-bearing. Said out
        // loud because a reader otherwise cannot tell that case from one where
        // forcing carried the whole comparison, and the two license different
        // claims: the first is comparable *without* this flag and the second is
        // comparable only *with* it.
        if forced_overrides == 0 {
            println!(
                "FORCED note: this arm sampled the forced sequence unaided, so its tokens would \
                 have matched the reference without forcing (the §10.2g case)"
            );
        }
    }

    // What the ICB executor actually replayed.
    //
    // The digest below is only evidence about replay if replay *happened*, and a
    // flag being passed does not establish that -- #69's determinism run under a
    // new hazard key reported a passing digest for the unchanged path, and was
    // caught by checking that a quantity had moved rather than by trusting the
    // flag (§2.4, §9.2f). A nonzero `covered` here is that quantity: it is the
    // number of dispatches this run did *not* encode classically.
    #[cfg(feature = "metal")]
    if let Some(exec) = icb_executor.as_ref() {
        let cov = exec.coverage();
        println!(
            "ICB replaying={} covered={} of {} runs={} barriers_encoded={} stale={} \
             poisoned_runs={} discarded_windows={} excluded_varies={} excluded_inline_consts={}",
            exec.is_replaying(),
            cov.covered,
            cov.positions,
            exec.runs(),
            exec.encoded_barriers(),
            exec.stale_positions(),
            exec.stale_runs(),
            exec.discarded_windows(),
            cov.varies,
            cov.inline_constants,
        );
        let stale = exec.stale_report();
        if !stale.first.is_empty() {
            let by_axis: Vec<String> = stale
                .by_axis
                .iter()
                .map(|(axis, n)| format!("{axis}={n}"))
                .collect();
            println!(
                "ICB stale: steps_with_stale={} prefix_flushes={} rerecordings={} by_axis: {}",
                stale.steps,
                exec.prefix_flushes(),
                exec.rerecordings(),
                by_axis.join(" ")
            );
            for (kernel, n) in &stale.by_kernel {
                println!("ICB stale {n:4} {kernel}");
            }
            for (step, pos, kernel, axis, at_head) in stale.first.iter().take(16) {
                println!(
                    "ICB stale first: step={step} pos={pos} {kernel} {axis} {}",
                    if *at_head { "head" } else { "member" }
                );
            }
        }
        println!("ICB barrier scope: {icb_barrier_scope:?}");
        // Issue #144. Printed beside the scope because the two axes are
        // independent and a reader comparing arms needs both: `BarrierScope`
        // governs the ordering the ICB *encodes*, `ReplayBarriers` the ordering
        // candle *still emits* beside it (§11.3n -- additive, not substitutive).
        //
        // `barriers_suppressed` is the quantity that shows the switch engaged,
        // which is the check #69's vacuous determinism run earned (§2.4, §9.2f):
        // both arms there silently ran the default and the "changed" arm
        // reported a passing digest for the unchanged path. **It is engagement
        // proof and never the correctness argument** -- a wrongly-suppressed
        // edge leaves this higher and the barrier count lower, both plausible.
        // The correctness evidence is this harness's own digest.
        println!(
            "ICB replay barriers: {icb_replay_barriers:?}  suppressed={}",
            candle::kernels::metal::trace::barriers_suppressed(),
        );
        // The residency term (`DESIGN.md` §11.3w): what one replayed step
        // declares, against what it would declare plan-wide, and what the
        // encoder actually declared over the run. `useResource` is counted at
        // the call site, so `per_step` here is a plan-time figure and the
        // process-wide total is the observed one.
        {
            use candle::kernels::metal::trace;
            let pool = candle::kernels::constants_pool()
                .map(|p| p.counters())
                .unwrap_or_default();
            // The arm is named from the plan, not from the variable that
            // selected it (§7.1a-i): per-run declares `per_step`, plan-wide
            // declares `runs x plan_wide`.
            let arm = if exec.residency_per_step() == exec.runs() * exec.plan_resident() {
                "plan-wide"
            } else {
                "per-run"
            };
            println!(
                "ICB residency arm={arm} per_step={} plan_wide={} runs={} execute_ranges={} \
                 useResource_total={} pool_allocated={} pool_unchanged={} pool_rewritten={} \
                 pool_detached={}",
                exec.residency_per_step(),
                exec.plan_resident(),
                exec.runs(),
                trace::execute_range_count(),
                trace::use_resource_count(),
                pool.allocated,
                pool.unchanged,
                pool.rewritten,
                pool.detached,
            );
        }
        // Per kernel, with the reason attached, because the totals above cannot
        // distinguish a position that became *covered* from one that merely
        // changed which bucket excludes it. #103 gave `sdpa_vector` a packed
        // sibling and `excluded_inline_consts` went 8 -> 0 while `covered`
        // stayed at 433: the 8 moved to `varies`. A reader comparing only the
        // totals would read the first half as a win and never see the second.
        for ((kernel, reason), n) in &cov.excluded_by_kernel {
            println!("ICB excluded {n:4} {kernel:<40} {reason}");
        }
        anyhow::ensure!(
            exec.is_replaying() && cov.covered > 0,
            "--icb was passed but nothing was replayed ({} covered of {}), so the digest below \
             would be the classical path's wearing this arm's name",
            cov.covered,
            cov.positions
        );
        anyhow::ensure!(
            exec.stale_positions() == 0,
            "{} replayed positions went stale, so coverage is lower than reported and the \
             digest describes a partially-classical run",
            exec.stale_positions()
        );
    }

    // One machine-greppable line per run, so a batch of runs collapses to a
    // sort|uniq over digests.
    //
    // There is deliberately no throughput field here, and `elapsed_ms` is not one
    // (lloom #102). This harness measures digests; `lfm2-decode-profile` measures
    // throughput, and only its figure is comparable to anything in `.bench/`.
    //
    // A `tok_per_s` was printed until 2026-08-26 and it was a whole-generation
    // average -- `generated / elapsed` -- so prefill, sampling, warmup and a
    // `kv_len` that grows across turns were all inside it.
    // `measurements/issue-7-reconciliation.md` §1 was written to discredit exactly
    // that shape, and it survived here for another twenty-odd issues and misled two
    // readers. Worked from one of its own runs: 267 tokens over 8272 ms reads
    // 32.28 tok/s; backing out three prefills at 90.22 ms leaves 29.97 ms/token
    // against §6.6's 18.763 ms steady-state decode, a factor of 1.60 that prefill
    // alone does not explain.
    //
    // The trap is not that the number is imprecise, it is that the two arms of a
    // determinism A/B generate *different numbers of tokens by construction*: when
    // the digests diverge the arms take different paths and stop at different
    // points (#97's were 267 and 401). So `elapsed_ms` is incomparable between arms
    // and any rate derived from it is incomparable too, sitting exactly where a
    // reader looks for a comparison.
    //
    // `elapsed_ms` stays because §2.3.8's evidence is that wall time varied while
    // the bits did not; it is a duration, not a rate, and dividing it by
    // `generated` reconstructs the discredited figure. Measured while removing the
    // field: every run taken for lloom #102 produced one digest pair, with wall
    // clocks spanning 8291 to 20085 ms as the machine's load moved -- a 2.4x
    // spread on the removed quantity. Sharpest inside a single three-run gate, one
    // binary, minutes apart: 31.90, 13.41 and 13.34 tok/s, from bit-identical
    // output -- and the preceding gate reproduced the same shape, so a 2.4x swing
    // within one gate happened twice out of two
    // (`measurements/issue-102-tok-per-s-removed.md` §4.1).
    //
    // `forced` and `overrides` are on this line because a digest is not
    // interpretable without them (issue #170), for the same reason §2.3.6 puts
    // the math mode here: under forcing the token digest is a digest of the
    // *forced* sequence, so two arms agreeing on it means only that both were
    // driven through the same ids. `overrides` is what says whether they would
    // have agreed on their own.
    println!(
        "RESULT label={} n={} tokens_sha256={} logits_sha256={} prompt_tokens={} generated={} \
         turns={} final_kv_len={} hit_eos={} dtype={:?} device={} seed={} temp={} top_p={} \
         forced={} overrides={} elapsed_ms={}",
        args.label,
        args.n,
        token_digest,
        logits_digest,
        prompt_ids.len(),
        tokens.len(),
        turns_run,
        kv_len,
        hit_eos,
        dtype,
        if args.cpu { "cpu" } else { "metal" },
        args.seed,
        args.temperature,
        args.top_p,
        forced_steps,
        forced_overrides,
        elapsed.as_millis(),
    );

    // Issue #559 (`DESIGN.md` §11.5n). On its own line rather than folded into
    // `RESULT`, for the reason the `POISON` line below gives: every committed
    // artifact's `RESULT` line has that shape and #171 records what widening it
    // costs an ingest of the existing corpus.
    //
    // **This line is what licenses reading the digests above as evidence about
    // the completion arm.** The digest gate is what says the async wait changes
    // how bytes move rather than what they are -- and it says that only if the
    // arm was live. An unmoved digest on an arm that never engaged is #500's
    // finding exactly: a mechanism with no callers regresses nothing, and the
    // clean board carries no information.
    //
    // So `completion_signals = 0` under an async arm is a **FAILURE TO ENGAGE**
    // rather than a clean run, and it is asserted rather than printed and
    // hoped over. `synchronize` is exempt because encoding nothing is precisely
    // what the incumbent must do -- the assertion is on the arm that claims to
    // do something.
    #[cfg(feature = "metal")]
    if let candle::Device::Metal(m) = &device {
        let arm = m.completion_wait_name();
        let (signals, polls) = m.completion_counts();
        println!("COMPLETION arm={arm} signals={signals} polls={polls}");
        anyhow::ensure!(
            arm == "synchronize" || signals > 0,
            "completion arm {arm} encoded no signal, so this run's unmoved digest \
             is vacuous -- the mechanism did not engage (§2.4, §15.1a, §11.5i)"
        );
    }

    // Issue #475. On its own line rather than folded into `RESULT`, for the
    // reason the `SPEC` line below gives: every committed artifact's `RESULT`
    // line has that shape and #171 records what widening it costs an ingest of
    // the existing corpus.
    //
    // **This line is what licenses a clean sanitizer run.** #475's registered
    // prediction is that ASan finds zero new defects on the decode path, and a
    // clean run proves that only if the mechanism was live -- otherwise it is
    // §15.1a's VACUOUS class, which is #69's consumed `OnceLock` and #454's
    // `to_cpu_fast=0` sweep. So the counters are printed whenever the feature
    // is compiled in, and `poisoned=0 unmappable=0` is a FAILURE TO ENGAGE
    // rather than a clean pool.
    //
    // `unmappable` is not an error and must be read beside `poisoned`: a
    // `StorageModePrivate` allocation at or above 16384 B has no CPU mapping,
    // so there is nothing for a poison to mark (§15.2 #12a, re-probed for
    // #475). It is the share of the pool the threshold puts outside the shadow
    // map, and reporting coverage without it invites reading an unmapped pool
    // as a verified one.
    {
        use candle::kernels::metal::poison;
        if poison::ENABLED {
            let c = poison::counters();
            println!(
                "POISON enabled=true poisoned={} unpoisoned={} unmappable={} empty={}",
                c.poisoned, c.unpoisoned, c.unmappable, c.empty,
            );
            anyhow::ensure!(
                c.poisoned > 0 || c.unmappable > 0,
                "the `asan-poison` feature is compiled in and the pool poisoned \
                 nothing and found nothing unmappable, so this run's clean \
                 verdict is vacuous -- the mechanism did not engage (§2.4, \
                 §15.1a)"
            );
        }
    }

    // Issue #89. On its own line rather than folded into `RESULT`, because every
    // committed artifact's `RESULT` line has that shape and #171 records what
    // widening it costs an ingest of the existing corpus.
    //
    // `windows` and `accepted/proposed` are the acceptance rate the cost model
    // needs beside a timing; `corrupted` is the wrong-proposer arm's engagement
    // proof (§2.4), since an arm named `wrong` that corrupted nothing would be
    // the oracle arm wearing the rollback arm's name.
    if let Some(k) = args.speculate {
        println!(
            "SPEC k={} proposer={} windows={} proposed={} accepted={} corrupted={} \
             accept_rate={:.4} tokens_per_window={:.4}",
            k,
            args.spec_proposer,
            spec_windows,
            spec_proposed,
            spec_accepted,
            spec_corrupted,
            if spec_proposed > 0 {
                spec_accepted as f64 / spec_proposed as f64
            } else {
                0.0
            },
            if spec_windows > 0 {
                spec_accepted as f64 / spec_windows as f64
            } else {
                0.0
            },
        );
        anyhow::ensure!(
            spec_windows > 0,
            "--speculate was passed and no verify window ran, so the digest above is the \
             ordinary decode path's result wearing the speculative arm's name (§2.4)"
        );
        if spec_wrong_every.is_some() {
            anyhow::ensure!(
                spec_corrupted > 0,
                "--spec-proposer wrong:<N> corrupted nothing, so this arm is the oracle \
                 arm and the rollback was never exercised"
            );
        }
    }

    // The raw stream, so the reported hash is independently checkable with
    // `shasum` rather than trusted.
    println!(
        "TOKENS {}",
        tokens
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::parse_forced_tokens;

    /// The `TOKENS` line this harness prints must feed straight back into
    /// `--force-tokens`, because that round trip is how a reference sequence is
    /// produced in practice (issue #170).
    #[test]
    fn the_tokens_line_round_trips_into_force_tokens() {
        let ids = parse_forced_tokens("785,267,3728,9560\n").unwrap();
        assert_eq!(ids, vec![785, 267, 3728, 9560]);
    }

    #[test]
    fn one_id_per_line_with_comments_and_blanks() {
        let ids = parse_forced_tokens("# a reference sequence\n785\n\n267  # trailing\n\t3728\n")
            .unwrap();
        assert_eq!(ids, vec![785, 267, 3728]);
    }

    /// A parse error is refused rather than skipped.
    ///
    /// Skipping would force a **different, shorter** sequence than the file
    /// names, and both arms would agree on it — a comparison that is internally
    /// consistent, reproducible, and not the one that was asked for. That is
    /// §2.4's shape exactly, and the failure would be invisible in the output
    /// because `forced` would still be nonzero.
    #[test]
    fn an_unparseable_entry_is_refused_not_skipped() {
        let err = parse_forced_tokens("785\nnot-a-token\n267\n").unwrap_err();
        assert!(
            format!("{err}").contains("not a token id"),
            "unexpected error: {err}"
        );
        // The load-bearing half: it did not quietly return [785, 267].
        assert!(parse_forced_tokens("785\nnot-a-token\n267\n").is_err());
    }

    #[test]
    fn an_empty_file_is_refused() {
        assert!(parse_forced_tokens("# only a comment\n\n").is_err());
    }

    /// A negative id is a parse failure rather than a wrap to `u32::MAX`.
    #[test]
    fn a_negative_id_is_refused() {
        assert!(parse_forced_tokens("785\n-1\n").is_err());
    }
}
