# lloom-engine: chat and local HTTP serving

`lloom-engine` combines the model/decode library with tokenization, checkpoint
chat templates, incremental output, prefix reuse and an HTTP frontend. The
`lloom` CLI requires the `serve` Cargo feature. The library alone need not pull
in the HTTP/tokenizer/template stack.

Start with [the supported-model/build overview](../README.md). This guide
names implemented interfaces, not full OpenAI API compatibility or validation
of every checkpoint/backend combination. Family and build capabilities govern
what a request can actually do.

## Start one local model

From the Cargo workspace (`candle/` in the full repository):

```sh
cargo build --release -p lloom-engine --bin lloom --features metal,serve
./target/release/lloom serve --model /path/to/LFM2.5-2.6B \
  --served-model-name local --host 127.0.0.1 --port 8080 --ctx-size 4096
```

For CPU, build with `--features serve` and use `--device cpu`. A compatible
LFM2 GGUF may be passed as `--model /path/to/model.gguf`; choose a context no
larger than its declared limit. A GGUF is not interchangeable with a
safetensors directory, nor is every GGUF architecture supported.

Wait for the actual stdout ready marker:

```text
{"kind":"ready","host":"127.0.0.1","port":8080}
```

Other operator messages go to stderr. With `--port 0` (the default), use the
port from the marker. `--host` defaults to `127.0.0.1`. A process holds one
loaded model: the request's `model` string does not load/switch checkpoints.
`--served-model-name` sets the id advertised by `/v1/models`.

The compatibility binary is the same frontend under another name:

```sh
cargo build --release -p lloom-engine --bin lloom-serve --features metal,serve
./target/release/lloom-serve --model /path/to/LFM2.5-2.6B --port 8080 --ctx-size 4096
```

There is no route-level API-key authentication or TLS. Keep loopback binding
unless a separately configured security boundary handles remote access. This is
not an untrusted multi-tenant service; `/shutdown` is administrative.

## HTTP routes

The normal generating listener implements these routes. A d1-omni
**decision-only** listener implements only `/health`, `/v1/models`,
`/v1/systemone`, and `/shutdown`; it returns 501 for generation routes.

| Method / path | Request and response scope |
|---|---|
| `GET /health` | Actual loaded capabilities, template/reasoning information, memory and reuse/diagnostic reports; owner-dependent fields |
| `GET /v1/models` | One served model and its declared input/decision capabilities |
| `POST /tokenize` | `{"prompt":"...","add_special_tokens":true}` → `tokens`, `count`; no chat-template rendering |
| `POST /v1/completions` | String or token-id prompt, generation fields, optional SSE; JSON `choices`, `usage`, `timings` |
| `POST /v1/chat/completions` | `messages`, checkpoint template, optional reasoning and SSE; JSON `choices`, `usage`, `timings` |
| `POST /v1/systemone` | Decision questions over supplied state, when the checkpoint/policy supports them |
| `POST /prefill_throughput` | `prompt_tokens` (and supported `batch`) → prompt throughput/time |
| `POST /decode_throughput` | `prompt_tokens`, `decode_tokens` → decode throughput, bracket and readback |
| `POST /end_to_end_latency` | `prompt`, `decode_tokens` → total time and counts |
| `POST /max_memory_usage` | `prompt_tokens`, `decode_tokens`; a benchmark driver can sample process memory |
| `POST /eval` | Benchmark JSONL events, not OpenAI chat SSE |
| `POST /eval/abort` | `sample_id` to stop an evaluation sample |
| `POST /shutdown` | Orderly shutdown |
| `GET /v1l/features` | The lloom extensions this process serves: the session fields and their values, the TTL and capacity, the chat extensions |
| `GET /v1l/sessions/{id}` | What the server holds of a session (turns, ids, age, expiry) — never its content |
| `DELETE /v1l/sessions/{id}` | Forget a session now |

There are no implemented `/v1/responses`, embeddings or files/assistants
routes in this listener; sessions are the chat route's (below). Template `tools` input does not
execute tools. Guided/constrained decoding fields such as `guided_choice` and
`guided_json` are refused, not silently enforced.

## JSON requests

```sh
curl -fsS http://127.0.0.1:8080/health
curl -fsS http://127.0.0.1:8080/v1/models
curl -fsS http://127.0.0.1:8080/tokenize \
  -H 'Content-Type: application/json' -d '{"prompt":"Hello","add_special_tokens":true}'

curl -fsS http://127.0.0.1:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local","prompt":"The capital of France is","max_tokens":16,"temperature":0}'

curl -fsS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local","messages":[{"role":"user","content":"Say hello."}],"max_tokens":64,"temperature":0,"return_token_ids":true}'
```

`max_tokens` limits new tokens; context includes prompt plus continuation.
`temperature: 0` is greedy. Sampling fields and `seed` select supported
samplers. `stop` and `ignore_eos` are available; suppressing EOS still needs a
finite generation budget. Already-tokenized chat uses `prompt_token_ids`
**instead of** `messages`; the caller owns the rendered prompt/special tokens.
`return_token_ids` exposes ids for reproducibility, separately from logprobs.

Chat `logprobs` is boolean, with `top_logprobs` 0–64; completions use their
completion-route shape. Availability depends on decode body, stream/media and
owner. Read `/health` and explicit refusals rather than assuming every route
can return a row. Token counts are not UTF-8 character counts.

## SSE and reasoning

```sh
curl -N -sS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local","messages":[{"role":"user","content":"Count to five."}],"max_tokens":64,"temperature":0,"stream":true}'
```

Consume SSE `data:` JSON events until the successful `[DONE]` terminator.
Accumulate `choices[].delta.content` and, when supplied,
`choices[].delta.reasoning_content` separately. Markers can cross fragment
boundaries and a budget/stop can end inside one. The final buffered partial
tail remains output. A failed transport is not a successful completed stream
and need not contain `[DONE]`. Server written-byte witnesses do not prove
receipt by a client.

Generating completions keep at most **eight response contexts**, including
requests waiting for their model turn and responses still draining. A ninth
request receives HTTP 503 (`response drain capacity exhausted`) before model
execution. The context is reclaimed only when its response writer finishes.
Each SSE response permits **1 MiB of unresolved raw text**, including the event
currently being written. Exceeding that backlog refuses the next whole fragment:
the earlier accepted frontier remains the history, `finish_reason` is
`cancelled`, and `timings.response_backlog_limit` is true. Typed reasoning
observations additionally permit 65,536 marker descriptors; exceeding that
limit fails the run rather than silently treating marker tokens as literal text.

The single-sequence model ticket is released after generation has returned its
request checkout, the accepted reply/session is published, and request-bound
metadata is captured — **before** socket draining. A continuation can therefore
use `previous_response_id` while the preceding response still drains. Stored
reply fields mean server-accepted text, not remotely acknowledged bytes;
`written_bytes` and `write_failed` are settled later by transport. No GPU
retirement certificate or additional global device wait is implied.

These are response-context/backlog limits, not a process-memory, request-body,
connection-thread, or wall-clock drain guarantee. An indefinitely stalled legacy
writer can retain one of the eight contexts. Explicit controlled transport
policies retain their separate I/O/cancellation limits; default listener read,
disposal and pipelining policies are not changed here. Bulk `/eval` retains its
existing exclusive-run/control-channel contract.

Thinking is template- and model-dependent. On a compatible checkpoint:

```json
{"model":"local","messages":[{"role":"user","content":"Explain your conclusion."}],"chat_template_kwargs":{"enable_thinking":true},"max_tokens":128,"stream":true}
```

`enable_thinking` must be boolean. `reasoning_effort` is a string in the
checkpoint template's own vocabulary, not a universal list. Top-level fields
and `chat_template_kwargs` must agree if both provide a value. A template
rejection is an error; a non-effective variable can be named in
`timings.template_kwargs_ignored`. Neutral/off semantics are resolved by the
frontend; arbitrary names do not create reasoning support. Inspect returned
`thinking`, `thinking_by`, `reasoning_effort` and prompt digest.

CLI chat has `--think`, `--no-think` and, where supported,
`--reasoning-effort VALUE`; it preflights unsupported/non-effective effort.
Serve uses request kwargs, not every chat-only flag.

## Conversations and history

Send full `messages` on each request, or continue a session by its id (below).
Send the **actual** preceding assistant message, including returned
`reasoning_content`, before the next user message. Python standard-library
example:

```python
import json
from urllib.request import Request, urlopen
url = "http://127.0.0.1:8080/v1/chat/completions"
messages = [{"role": "user", "content": "Name a blue boat."}]
def turn():
    body = json.dumps({"model": "local", "messages": messages,
                       "temperature": 0, "max_tokens": 64}).encode()
    req = Request(url, body, {"Content-Type": "application/json"})
    with urlopen(req, timeout=120) as response:
        return json.load(response)
first = turn()
messages.append(first["choices"][0]["message"])
messages.append({"role": "user", "content": "Use that name in a sentence."})
print(turn()["choices"][0]["message"])
```

**Sessions** (`design/docs/sessions.md`). Every chat completion's `id` names
the session its reply left (`chatcmpl-` and 40 hex digits). `store: true` keeps
it until the server reclaims it — unused for `server.sessions.idle_s` (3600 s),
or the least recently used past `session.capacity` (64) or `session.max_mib`
(64) — or the client lets it go (`reset`, `DELETE /v1l/sessions/{id}`); without
it the id lasts while the engine still holds that conversation. A session has
no expiry of its own: the policy is the server's, reported on `/health`. To
continue:

- `previous_response_id` with `input` (a string, or the new messages): appended
  to the stored conversation; the history is neither sent nor tokenized again.
  An id the server does not hold is a 404 — resend the whole history.
- `previous_response_id` with `messages` (the whole history, as always): the
  session's replies are bound by the conversation before them, not by their
  text, so a reply the client restructured (tool calls parsed out of it) is
  still the model's; an id the server does not hold falls back to today's path.
- `reset: cache|prefix|all` after a compaction; `keep: none` (`prefix` and `all`
  need the engine's checkpoint pin and are refused by name in this build).

`timings.session` reports `{id, state: new|resident|evicted|expired, reason,
turn, switches, reprefilled_tokens, prefix, history, recorded_tokens,
tokenized}`, where `reason` says why an `expired` session was let go — `idle`,
`capacity`, `memory`, `reset`, `delete`, or `restart` for an id from an earlier
server process — and is `null` where the id's holder cannot be told (another
`Authorization`, a release too old to remember); a 404 for a session the server
let go names the reason too. `/health`'s `sessions` carries the policy
(`held`, `capacity`, `max_mib`, `idle_s`, `reclaim`) and what it let go, by
reason (`released`). An id is a bearer token: a client that sends
`Authorization` binds its sessions to that header's digest.

```python
first = post({"messages": [{"role": "user", "content": "Name a blue boat."}],
              "store": True, "max_tokens": 64})
second = post({"previous_response_id": first["id"],
               "input": "Use that name in a sentence.", "max_tokens": 64})
```

Serve prefix reuse defaults on (`--no-reuse` opts out); benchmark cells retain
their cold brackets. Reuse is not durable storage or authenticated tenant
isolation. `timings.turn` and `/health` say what actually ran.

Where `Served::text` supports reply splicing, the frontend can retain accepted
sampled ids/reasoning. `session.persist_thinking`, `session.missing_reasoning`
and `session.incomplete_reasoning` control that behavior. Incomplete reasoning
defaults to the `reasoning` policy, not a claim that truncation completed it.
These frontend policies are not a public clone/rollback guarantee for every
cache.

**LFM2 GGUF limitation:** its generic served owner does not expose the text
side to this history log. It renders ordinary full-history requests and serves
JSON/SSE, but `timings.logged=false` / an unpopulated accepted frontier is not
accepted-id history logging. Do not promise the safetensors owner's exact
sampled-id/reasoning splicing here. Cancellation may accept only a delivered
prefix; client receipt and server-written frontier are distinct.

## Images and decisions

Compatible LFM2-VL and native d1-3B support images when their tower is loaded.
GGUF requires the explicit matching `--mmproj`; no sidecar variant is guessed.
`--text-only` omits the tower and image requests refuse. Chat `image_url`
data URLs support the implemented image codecs, not arbitrary remote URL
fetching, video or Qwen vision. `/v1/models` reports loaded modalities.

`/v1/systemone` is a decision contract, not free-form generation. Native d1
metadata selects a binding; otherwise an explicit supported decision policy is
needed. Generic label scoring is uncalibrated, not native d1 calibration.
Example for an already loaded decision-capable checkpoint:

```sh
curl -fsS http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"state":"The sky is blue.","questions":{"color":{"type":"choice","instructions":"Which color is stated?","criteria":{"blue":"Blue","red":"Red"}}}}'
```

Questions are ordered named objects of type `choice`, `score` or `noul`, with
type-specific criteria/checkpoint limits. `state` must be present; nullability
is policy-dependent. d1-omni is non-generating with its own media/companion
requirements; ordinary chat is not its API. The full repository's
`design/docs/models/d1.md` records the decision contract/evidence.

## Concurrency, resources and diagnostics

Default `--batch 1` serializes generating requests. LFM2 and Qwen4-exp have
family-specific batching paths; Qwen3.5 and generic LFM2 GGUF refuse batching.
Speculative batching differs from the continuous K=1 scheduler. An off-token
writer keeps socket I/O away from decode; transport cancellation is distinct
from native completion and memory retirement.

**Keep-warm.** macOS unwires the GPU's memory a few seconds after its last
use, and the next request then waits while it is wired back. On Metal at
`--batch 1`, `serve` asks for it to stay resident every second while it idles
(`server.keep_warm`, default 1000 ms), for 180 s after the last request
(`server.keep_warm_for_s`). Then the heartbeat stops, an idle server gives the
memory back, and the next request starts it again. `--keep-warm 0` turns it
off; `--keep-warm-for 0` removes the bound; `--keep-warm-by dispatch` runs one
one-element dispatch instead, which keeps less of the memory wired. The loaded
line says the resolved policy (`keep_warm=1000ms/180s/residency`, or
`keep_warm=off`), a `[keep-warm: …]` line says it in words (and why, when
off), and `/health` carries `keep_warm`: the policy, whether the heartbeat is
running, its actions and how many windows ran out. Asking for it at
`--batch N` is refused (not built: the batch thread owns the device); on a
device that is not a GPU it is off. `lloom chat -p` and `lloom bench` run in a
fresh process and do not idle between load and work.

`--ctx-size`, `--prefill-tile`, batch width, KV precision, resident weights and
streamed slots affect memory. Machine/backend admission is not a hard process
quota or an OOM guarantee. Reduce context/batch/working set on refusal; do not
use `--force` as the default remedy. Expert streaming still needs files and
I/O. CPU is an execution path; CUDA compilation is not blanket served-model
runtime validation. `auto` may fall back to CPU; inspect actual startup output.

```sh
./target/release/lloom config --frontend serve --ctx-size 4096 --explain
./target/release/lloom options --compiled
./target/release/lloom options --schema
```

Defaults < env < config < group JSON < `--set` < explicit flags; frontend
fallbacks apply to unnamed keys. A schema projection does not promise identical
native-number/string grammar. Reserved/unavailable keys can be listed without
being activated.

For an optional optimized diagnostic build:

```sh
cargo build --profile release-with-debug -p lloom-engine --bin lloom \
  --features metal,serve,profile
./target/release-with-debug/lloom options --compiled
./target/release-with-debug/lloom chat --model /path/to/LFM2.5-2.6B \
  --ctx-size 4096 -v -p 'Say hello.'
```

Symbols, profile collection and DWARF line material are distinct. Do not infer
complete detailed-recorder compile-out solely from `--release`; that diagnostics
cleanup is separate pending work, not a guarantee of this dev snapshot. Missing
telemetry is not a real zero. Diagnostic runs perturb timing; mandatory atomics,
accounting and safety checks are not optional profiling features.

Benchmark brackets differ: fresh-cache prefill includes its drain; ordinary
decode excludes prefill; speculative paths declare their own bracket;
end-to-end includes frontend work. `timings` is the engine bracket, not the
HTTP client's wall time. Memory sampling is a gauge run. Keep exact binary,
input, backend, math mode and seed beside performance numbers.

## Library integration

The existing library seam remains separate from HTTP:

```rust
use lloom_engine::{DecodeParams, DecodeSession, ReadbackArm};
// Caller supplies model, cache, device, prompt_ids, detokenize and actual EOS.
let mut session = DecodeSession::new(&model, cache, device, 0);
let params = DecodeParams {
    max_tokens: 256,
    eos_ids: checkpoint_eos_ids,
    readback: ReadbackArm::Auto,
    ..DecodeParams::default()
};
let out = session.decode(&prompt_ids, detokenize, &params,
                         |fragment| print!("{fragment}"))?;
```

This is an integration sketch, not a standalone Rust program or generic native
admission for custom callbacks. `decode_until` supports fragment-hook
cancellation; `reset` discards reusable session state. Slot/Completer library
mechanisms do not imply additional REST session routes.
