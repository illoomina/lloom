# lloom-engine

The serving engine over `lloom`'s tier-1 API (`RESTRUCTURING_DESIGN.md` §4,
§7.1): the decode loop with the serving concerns around it — incremental
detokenization, stop strings, cancellation, content-addressed prefix reuse —
and one binary, **`lloom-serve`**, that puts the engine on a loopback port.

## The library

```rust
use lloom_engine::{DecodeParams, DecodeSession, ReadbackArm};

let mut session = DecodeSession::new(&model, cache, device, 0);
let params = DecodeParams {
    max_tokens: 256,
    eos_ids: vec![124900],            // <|im_end|>
    readback: ReadbackArm::Auto,      // the crown ring where the sampler admits it
    ..DecodeParams::default()
};
let out = session.decode(&prompt_ids, detokenize, &params, |fragment| print!("{fragment}"))?;
// out.text, out.tokens, out.stop, out.stats (the loop's own clocks), out.readback (the arm that ran)
```

- `DecodeParams::readback` selects the readback arm (`design/docs/decode-driver-design.md`,
  seam A). `Auto` takes the device-resident token ring with one forward of
  lookahead — the crown — when the sampler is argmax
  or plain temperature, and the blocking host-readback arm for top-p. `Blocking`
  and `Ring { lookahead, commit_steps }` select explicitly.
- `decode_until` is `decode` whose fragment hook can end the run (`false` →
  `Stop::Cancelled`): a client that went away, a benchmark's early abort.
- A request is the **whole** token sequence; `Session::prepare` reuses the
  prefix the session's cache already holds. `DecodeSession::reset` discards it
  for a caller that must start cold.
- `SlotPool` routes requests to sequences by longest common prefix
  (llama.cpp's `get_available_slot`); `Completer` is the `dyn`-safe per-request
  seam.

## `lloom serve` / `lloom chat` (and `lloom-serve`)

```sh
cd candle
cargo build --release --features metal,serve -p lloom-engine --bins
./target/release/lloom serve --port 8080                 # LFM2.5-2.6B from the HF cache
./target/release/lloom serve --model ~/.cache/huggingface/hub/models--ISTA-DASLab--Qwen3.8-Flash-Next-GSQ-RCO-GGUF/snapshots/<hash>/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
./target/release/lloom chat --model <dir-or-gguf>        # a streaming REPL on the same engine
./target/release/lloom-serve --port 8080                 # `lloom serve`, under the name the pipette runner spawns
```

One binary over every served family (`src/app/`). `--model` names the
checkpoint and its family is read from the path: a safetensors directory
(`config.json`, `tokenizer.json`, `*.safetensors`, `chat_template.jinja`) is
**LFM2** — every arm the serving binary had (the crown ring, `--dspark`,
`--batch N`, the pool rows), byte for byte — and a GGUF shard (or a directory
of shards) is **Qwen4-exp** (`lloom::models::qwen4exp`): one sequence over
the **streamed** expert tier by default (`--experts stream --slots auto
--loader pread`, `design/docs/expert-streaming.md` §20; `--experts resident`
holds every expert in memory), the N-gram PLE on (`--ple-cache-rows N` for
its hot-row cache), `--mtp <gguf>` for the MTP head on the speculative body
(one drafted token per round, verified by the target, identical to greedy).
The tokenizer and the chat template come from the HF cache's
`Qwen/Qwen3.8-Flash-Next` snapshot (`--tokenizer` to point elsewhere).

**The template's variables** (`src/app/template.rs`). `lloom chat --think`
sets `enable_thinking`; `--reasoning-effort <EFFORT>` sets `reasoning_effort`
(Qwen4-exp: `low`, `medium`, `xhigh`; its default, and a run with no flag
renders byte-identically to `xhigh`). Which family supports a variable is
read from its template — a static parse of the source plus a render probe —
not from a list: `chat` refuses, before the first turn, an effort the template
does not read (LFM2), one it renders the same way for any value (Qwen4-exp
without `--think`) and a value it rejects (its own `raise_exception`
message); `serve` answers the last with a 400 and serves the first two
without the effort, naming it in `timings.template_kwargs_ignored`. The turn
line and `timings` carry `reasoning_effort` and `prompt_ids_sha256`, so a run
says which prompt it ran.

The request seam is `app::served::Served`, `dyn` per request (§15.2 #9); the
per-token loop behind it is `DecodeSession`'s, generic over `CausalLM`. On
Qwen4-exp `--readback auto` resolves to the **blocking** arm: the model's
`forward` needs its input ids on the host (the PLE hashes them), which the
blocking arm holds and hands over through `CausalLM::note_input_ids` — no
readback in the model; a ring arm is honoured by name and reads the fed id
back per step (correct, no faster). `lloom chat` and `lloom serve` run LFM2
on the same default, the flow: the adaptive commit, the ring reader, and the
EOS gate, whose forwards past a turn's EOS write nothing, so the turn's cache
ends at its EOS and the next turn extends it (`Session::prepare` → `Extend`)
instead of re-prefilling the conversation (`design/docs/dispatch.md`
§11.5z-xiv). `--readback ring-la0` (the ring without the lookahead, `lloom
chat`'s default before), `--readback ring-fixed` and `--eos-gate=false` stay
selectable; the gate is off, named at load, wherever the adaptive commit does
not run and under `--replay`.

### The run reports, and `lloom bench --mode`

`lloom` is also the measurement harness the retired `lfm2-chat` example was
(`src/app/report.rs`, `src/app/modes.rs`; the flag-for-flag map is
`design/docs/decode-driver-design.md` Stage 7):

```sh
lloom chat --raw -p "$PROMPT" -v                     # the turn's diagnostics: first token, cadence, churn, KV, …
lloom chat --raw -p "$PROMPT" --replay               # + what the ICB executor replayed (arena, packed, pass plans alike)
lloom chat --raw -p "$PROMPT" --dspark --phase-sync  # + the round breakdown and its DSPARK_ROUNDS line
CANDLE_METAL_PROFILE=1 lloom chat --raw -p "$PROMPT" # + GPU time by kernel, busy/wall, the idle between buffers
lloom bench --mode concurrent --batched --sequences 4 --stagger 8   # solo vs the batched driver, each stream checked
lloom bench --mode concurrent --dspark --sequences 2 --timeline r.csv  # solo vs the pipelined speculative driver
lloom bench --mode reuse                             # turn 2 through prefix reuse vs a fresh prefill
```

The modes are LFM2's (`Served::run_mode`; Qwen4-exp refuses by name): each
drives several sequences of one model directly, carries its solo reference
arm and prints one `RESULT` line on stdout.

Loads the LFM2.5-2.6B snapshot from the HF cache when no `--model` is given
(`LFM2_MODEL_DIR` to point elsewhere), warms the Metal pipelines, and serves one
request at a time on the crown decode configuration (`Sdpa × InPlace ×
Shuffle`, the GEMV epilogue and conv fusions — `--cache crown`'s arms, every
one token-identical to the generic path).

Two contracts on the one port:

| endpoint | shape | for |
|---|---|---|
| `GET /health` | `{}`; `chat_template_reads` says which of the engine's template variables (`enable_thinking`, `reasoning_effort`) the checkpoint's template reads; `logprobs` whether per-token logprobs are served (and why not); `reuse` every request's prefix reuse counted by outcome, reason and state, with the last request's report | readiness |
| `GET /v1/models` | OpenAI | model id |
| `POST /tokenize` | `{prompt, add_special_tokens?}` → `{tokens, count}` (also `{content, add_special}`) | exact-count prompt construction; counts what `/v1/completions` will report |
| `POST /v1/completions` | OpenAI + `ignore_eos`; text or token-id prompt; `stream`; `usage`; `timings` | pipette's `end_to_end_latency` / `max_memory_usage` cells, generic clients |
| `POST /v1/chat/completions` | OpenAI; the checkpoint's own `chat_template.jinja` (with `chat_template_kwargs`); `reasoning_effort` (OpenAI's field, or `chat_template_kwargs.reasoning_effort`; the values are the template's — Qwen4-exp: `low`, `medium`, `xhigh`); a value the template rejects → 400 with its message; an effort the template does not act on (LFM2, or thinking off) is served without it, with a warning; `timings` carries `reasoning_effort`, `template_kwargs_ignored` and `prompt_ids_sha256`; `stream` SSE; `logprobs` (`top_logprobs` up to 64, refused by name where the decode body reads no row); `return_token_ids`; `guided_choice` → 400 | pipette's `eval` cells, chat clients |
| `POST /prefill_throughput` | `{prompt_tokens}` → `{prompt_tps, prompt_tokens, prefill_ms}` | the MLX-sidecar contract pipette's throughput runner speaks |
| `POST /decode_throughput` | `{prompt_tokens, decode_tokens}` → `{generation_tps, decode_tokens, decode_ms, context_ms, readback}` | " |
| `POST /end_to_end_latency` | `{prompt, decode_tokens}` → `{total_ms, prompt_tokens, completion_tokens}` | " |
| `POST /max_memory_usage` | `{prompt_tokens, decode_tokens}` → `{prompt_tokens, completion_tokens}` | " (the runner reads `phys_footprint` on this process) |
| `POST /eval` | `{samples, max_tokens, temperature, completions_done_ids, enable_thinking}` → JSONL `eval_sample_start` / `eval_sample_chunk` / `eval_sample_done` / `eval_done` | " |
| `POST /eval/abort` | `{sample_id}` | early stop of a degenerate sample (doom-loop) |
| `POST /shutdown` | `{}` | orderly exit |

What the benchmark endpoints bracket, stated so the numbers are readable
(`design/docs/architecture.md` §2.5):

- `/prefill_throughput`: one forward over `prompt_tokens` seed tokens on a fresh
  cache, submitted and drained (`synchronize()`); nothing else.
- `/decode_throughput`: the prompt is prefilled and its first token sampled
  outside the bracket and the device drained; then `decode_tokens` tokens —
  `D` forwards and `D` samples on the readback arm named in the response — are
  timed to the last one landing. `llama-bench --n-depth P --n-gen D`'s shape.
- `/end_to_end_latency` and `/v1/completions`: tokenization, prefill and the
  decode inside the window, EOS suppressed when asked (`ignore_eos`), the exact
  counts in `usage`. The runner's own HTTP bracket is the published figure; the
  server's `total_ms` is the same window one layer in.
- `/v1/*` responses carry a `timings` block from the loop's own clocks:
  `prompt_submit_ms` (encode, not execution), `predicted_ms` (the sum of the
  loop's steps), the readback arm, commits, poll fallbacks, overrun, and
  `turn` — what the request reused of the session (the boundary it found,
  how the state there was made ready, the outcome and its reason, the ids
  kept, replayed and newly prefilled; `design/docs/sequence-reuse-design.md`
  §2.3f).
- A response is text. The session keeps the sampled ids internally (the
  next turn runs them), and returns ids — `prompt_token_ids`, `token_ids`,
  and `timings.tokens` for a gate's digest — only under `return_token_ids`,
  and logprobs only under `logprobs` (§2.3d).

The throughput cells tile a seed passage into their prompts: `--prompt-seed-file`,
else `PIPETTE_PROMPT_SEED_TEXT` / `PIPETTE_MLX_PROMPT_SEED_TEXT` (what the MLX
runner exports), else a built-in paragraph.

Flags: `--kv-capacity` (positions per request, default 8704 = the ladder's
8192 + 256 + slack; a request past it is refused, not reallocated),
`--readback auto|ring|blocking`, `--reuse-prefix` (off: a benchmark measures a
cold prefill), `--no-warmup`, `--cpu`, `--model-name`, the two below, and the
memory guards:

- `--prefill-tile` (default **512**, every path): the most prompt tokens one
  prefill forward takes. A prefill's activation working set is bounded by the
  tile — two tiles' worth, ~4.5 GB at 512 — rather than by the prompt; one
  forward over the whole prompt is ~3.9 MB of `pending` activation per token
  (37.8 GB at 8192 measured, and a 32k one-shot cell is what panicked this
  machine, #148). Every multi-tile prefill is paced against the GPU on every
  readback arm. `0` is the one-shot, admitted only if the budget below allows
  the KV ceiling's worth of activations.
- `--memory-fraction` (default 0.65; `0` = off): `MemoryBudget` admission —
  the weights, the KV ceiling's cache, the tile's activations and the scratch
  are predicted at load against `recommendedMaxWorkingSetSize` and refused with
  the reductions listed if they do not fit; a KV grow past the fraction is
  refused at the grow.
- `--mem-sample`: hold the memory gauges' peaks over every benchmark cell at
  10 ms and put them on the response (`mem`: `device_allocated`, the pool's
  live / free / `pending`, `phys_footprint`, `recommendedMaxWorkingSetSize`,
  before / peak / after). A gauge run, not a timing run. `GET /health` reads the
  gauges once at any time.

### `--batch N` — continuous batching (issue #148)

`--batch 1` (the default) is the shape above: one sequence, one request at a
time, the crown loop on the request's thread. `--batch N` starts the
continuous-batching scheduler (`lloom_engine::batch`) on its own thread over a
pool of `N` sequences: requests from any number of clients join the running
batch at the next step and leave it at their EOS — every live row's next token
and every arriving prompt's next chunk ride ONE `m = B` weight sweep per step
(`BatchedCausalLM::forward_ragged`, the CHUNK of #728) — so `N` concurrent
`/v1/completions` decode at the batched aggregate rather than queueing. A row's
token stream is the batched driver's (a device argmax over `[B, vocab]` and a
staged readback per step, `readback: "batched rows=k"`); a row with a
temperature is sampled on the host by its own seeded sampler, so its draw does
not depend on its neighbours. Prefix reuse across requests is `SlotPool`'s
routing (`--reuse-prefix`).

The benchmark cells take a `batch` field (`1` by default, at most `N`):

| endpoint | `batch` > 1 runs |
|---|---|
| `/prefill_throughput {prompt_tokens, batch}` | `batch` prompts of `prompt_tokens` each as ONE ragged forward (tile-bounded pieces); `prompt_tps` is the aggregate over `batch × prompt_tokens` |
| `/decode_throughput {prompt_tokens, decode_tokens, batch}` | every row's context built outside the bracket, then `decode_tokens` batched steps timed together; `generation_tps` is the aggregate, `per_row_tps` its share; `bracket` says what the window holds |
| `/end_to_end_latency {prompt, decode_tokens, batch}` | the same request `batch` times at once through the request path; `total_ms` closes when the last lands, `per_row_ms` carries each |
| `/max_memory_usage {prompt_tokens, decode_tokens, batch}` | `batch` concurrent requests resident at once |
| `/eval` | the samples `N` at a time, `N` workers pulling from the list; events interleave on the stream keyed by `sample_id` |

These closed cells run on the scheduler thread with nothing else live (the
engine to themselves, as at `--batch 1`). `--prefill-tile` and
`--prefill-chunk` (both 512 at `N > 1`, 0 = unbounded / whole) are the batched
steps' prefill bounds (#728).

### `--dspark` — the speculative body

Loads the LFM2.5-2.6B-DSpark drafter (`--draft-dir`, or the newest snapshot in
the HF cache) and serves every request through `Session::decode_speculative`
(`design/docs/speculation.md` §10.2t): a block of `K = 9` proposals per round,
verified by the target in one pass, `readback: "dspark k=9"`. Greedy by
construction — a `temperature` is refused, not served as argmax. With
`--batch N` the scheduler runs the rows in STATIC batches through the pipelined
driver (`decode_speculative_pipelined`, §10.2y): the waiting requests up to
`N` at once, to completion, then the next — a row cannot join mid-batch there.
The `/decode_throughput` bracket under `--dspark` is the run's wall minus the
prompt's submit (the speculative body holds no resumable logits to split the
context from the decode); the response's `bracket` field says so.

Two things a request never pays for (#702): the token ring — 32 MiB of
consumer slack, ~35 ms to build — is built **once per EOS set at load**
(`[rings: 2 built at load …]`; `RingPool` in `decode.rs`, keyed on the EOS
set because the ring uploads it once) and reused by every run, its history
wrapping (a ring's capacity is its live window, not the generation); and the
streaming writes (`/v1/*` SSE, `/eval` JSONL) go through a **writer thread**
(`StreamWriter` in `main.rs`), so the thread running the decode never blocks
on the socket — a client that stops reading stalls the queue, not the GPU,
and one that goes away stops the run at the next drained batch.
