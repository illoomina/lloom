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
  lookahead — the arm `lfm2-chat` calls the crown — when the sampler is argmax
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

## `lloom-serve`

```sh
cd candle
cargo build --release --features metal,serve -p lloom-engine --bin lloom-serve
./target/release/lloom-serve --port 8080          # prints {"kind":"ready","host":"127.0.0.1","port":8080} on stdout
```

Loads the LFM2.5-2.6B snapshot from the HF cache (`--model-dir` /
`LFM2_MODEL_DIR` to point elsewhere), warms the Metal pipelines, and serves one
request at a time on the crown decode configuration (`Sdpa × InPlace ×
Shuffle`, the GEMV epilogue and conv fusions — `lfm2-chat --cache crown`'s
arms, every one token-identical to the generic path).

Two contracts on the one port:

| endpoint | shape | for |
|---|---|---|
| `GET /health` | `{}` | readiness |
| `GET /v1/models` | OpenAI | model id |
| `POST /tokenize` | `{prompt, add_special_tokens?}` → `{tokens, count}` (also `{content, add_special}`) | exact-count prompt construction; counts what `/v1/completions` will report |
| `POST /v1/completions` | OpenAI + `ignore_eos`; text or token-id prompt; `stream`; `usage`; `timings` | pipette's `end_to_end_latency` / `max_memory_usage` cells, generic clients |
| `POST /v1/chat/completions` | OpenAI; the checkpoint's own `chat_template.jinja` (with `chat_template_kwargs`); `stream` SSE; `logprobs`/`top_logprobs` at `max_tokens = 1`; `guided_choice` → 400 | pipette's `eval` cells, chat clients |
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
  loop's steps), the readback arm, commits, poll fallbacks, overrun.

The throughput cells tile a seed passage into their prompts: `--prompt-seed-file`,
else `PIPETTE_PROMPT_SEED_TEXT` / `PIPETTE_MLX_PROMPT_SEED_TEXT` (what the MLX
runner exports), else a built-in paragraph.

Flags: `--kv-capacity` (positions per request, default 8704 = the ladder's
8192 + 256 + slack; a request past it is refused, not reallocated),
`--readback auto|ring|blocking`, `--reuse-prefix` (off: a benchmark measures a
cold prefill), `--no-warmup`, `--cpu`, `--model-name`, and the two below.

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
