<div align="center">

<img src="assets/lloom-logo.jpg" alt="lloom" width="360" />

**Local LLM inference, blazing fast.**

</div>

lloom is a hard fork of [Candle](https://github.com/huggingface/candle), with
specialized Metal kernels, speculative decoding, and an HTTP serving engine.
CPU execution is also available. The same `lloom` executable runs interactive
chat, serves requests, and exposes its resolved configuration.

> **Preview software.** Model support is specific to the loaders below, not
> every model supported by upstream Candle or every file named `.gguf`.
> Interfaces and options can change. This is a snapshot of implemented routes
> and their limits, not a claim that every checkpoint/backend combination has
> been validated. Unfinished paid-pool, batch-ownership, and session/recovery
> work is not a shipped guarantee.

## Build

Use a current Rust toolchain and a working native linker. Metal builds require
Apple silicon, macOS, and the Xcode command-line/Metal development tools.

```sh
# Apple silicon / Metal
cargo build --release -p lloom-engine --bin lloom --features metal,serve

# CPU-only, including Linux; no Metal dependency
cargo build --release -p lloom-engine --bin lloom --features serve
```

Both commands produce `target/release/lloom`; building the second in the same
target directory replaces that executable. `serve` is needed for the CLI,
including `chat`; it does not select a GPU backend.

| Feature | Purpose / limit |
|---|---|
| `serve` | `lloom` and the compatibility binary `lloom-serve`; HTTP, tokenizer, and chat-template dependencies |
| `metal` | Metal backend; primary optimization and native validation platform |
| `cuda` | CUDA backend build path; requires a matching CUDA development toolchain. Compilation is not evidence that every served model/quantization runs on CUDA |
| `accelerate`, `mkl` | Optional CPU library backends; require the corresponding platform/toolchain |
| `profile` | Detailed instrumentation; use the separate diagnostic build below, not throughput measurements |

`--device auto` tries Metal, then CUDA if compiled, then CPU. `--device cpu`
forces CPU. Inspect the actual startup device/capability lines rather than
assuming a GPU was selected. An explicit unavailable Metal4 request is refused,
not silently converted to CPU.

## Choose a local checkpoint

`--model` is a **local path**, not a Hub repository id. The server loaders read
existing files; they do not download missing checkpoints. Supply the matching
tokenizer/template and every required shard. Use the checkpoint's license and
instructions when obtaining it separately.

| Served family | Checkpoint input | Implemented surface / important limits |
|---|---|---|
| LFM2 / LFM2.5 text | Safetensors directory: `config.json`, `tokenizer.json`, weight files; `chat_template.jinja` for chat | Chat/completions; LFM2 DSpark with a compatible drafter; family-specific batching and cache arms |
| LFM2-VL / native d1-3B | Compatible LFM2-VL safetensors directory | Image input with the loaded tower; native d1 decision binding where checkpoint metadata identifies it |
| LFM2 GGUF | GGUF with `general.architecture=lfm2` | Text chat/completions, including quantized weights; one sequence, no DSpark/MTP or integer-quantized KV. Matching `--mmproj` enables supported vision; `--text-only` disables it |
| Qwen4-exp / Flash-Next | Matching GGUF shard set, `general.architecture=qwen4exp` | Streamed or resident experts; MTP/DFlash where the required drafter is available. No vision tower. Not a generic Qwen GGUF loader |
| Qwen3.5-family text | GGUF `qwen35`, `qwen35moe`, or compatible `clef` text architecture | Dense or resident-MoE text; optional MTP when the file/sidecar supplies it. One sequence; no vision; served KV is F32 |
| d1-omni | Recognized `d1_omni` safetensors or native decision GGUF plus required media companion | **Decision-only** `/v1/systemone`; not autoregressive chat/completions |

Quantized **weights** and quantized **KV cache** are separate capabilities.
LFM2 GGUF can use its packed weight formats (for example Q8_0) with F32 or F16
KV; that does not enable Int8/Int4 KV. Qwen4-exp reduced KV stores are
single-sequence capabilities; its batched paged pool uses F32. LFM2 safetensors
has its own reduced-KV arms and backend restrictions. The loader rejects
incompatible shapes/options rather than making all matrix cells equivalent.

LFM2 GGUF uses `--tokenizer` if given, then `tokenizer.json` beside the GGUF,
then the embedded vocabulary; its embedded chat template must be usable for
chat. Qwen3.5 can use sidecar/base-snapshot or embedded tokenizer/template
metadata. Qwen4-exp needs a matching external tokenizer/template (pass
`--tokenizer /path/to/tokenizer.json`, with `chat_template.jinja` beside it,
or provide its expected cached snapshot). All shards of one model belong
together; select the first shard or its dedicated shard directory.

Without `--model`, the loader searches the local default LFM2.5-2.6B snapshot
under `~/.cache/huggingface/hub/`; `LFM2_MODEL_DIR` is the legacy explicit path
override. Explicit paths are preferable for reproducibility and custom caches.

## Chat

```sh
# A safetensors checkpoint directory
./target/release/lloom chat --model /path/to/LFM2.5-2.6B \
  --ctx-size 4096 --temperature 0 -p 'Explain how a hash map works.'

# A real local GGUF on CPU; choose a context no larger than its declared limit
./target/release/lloom chat --model /path/to/LFM2.5-2.6B-Q8_0.gguf \
  --device cpu --ctx-size 4096 --temperature 0 -p 'Say hello.'

# Omit -p for the interactive, multi-turn REPL
./target/release/lloom chat --model /path/to/LFM2.5-2.6B --ctx-size 4096
```

The REPL applies the checkpoint's chat template and carries conversation
history. `--raw -p TEXT` instead encodes an already-rendered prompt. For a
compatible safetensors LFM2 target and drafter, `--dspark --draft-dir /path/to/drafter`
selects speculation; this is not a switch for arbitrary GGUFs. Unsupported
methods/combinations are refused. Use `lloom chat --help` for the exact build's
flags, including image and reasoning controls.

## Serve

```sh
./target/release/lloom serve --model /path/to/LFM2.5-2.6B \
  --served-model-name local --host 127.0.0.1 --port 8080 --ctx-size 4096
```

Wait for the actual stdout marker `{"kind":"ready",...}` before making
requests. Startup loads one checkpoint; the request's `model` field does not
load or switch models. The default host is loopback; the default port is `0`
(an automatically selected port, reported in the marker). Pin a port for scripts.

```sh
curl -fsS http://127.0.0.1:8080/health
curl -fsS http://127.0.0.1:8080/v1/models

curl -fsS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local","messages":[{"role":"user","content":"Say hello."}],"max_tokens":64,"temperature":0}'

# SSE: -N disables curl output buffering
curl -N -sS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local","messages":[{"role":"user","content":"Count to five."}],"max_tokens":64,"temperature":0,"stream":true}'
```

Successful streaming uses `data: <JSON>` events and a final `data: [DONE]`.
Chat fragments may contain `delta.content` or, when the checkpoint actually
uses thinking markers, `delta.reasoning_content`. Consume both; reasoning is
not a universal model capability. A broken stream is not a complete successful
response. The server's written-byte/frontier diagnostics describe what it knows
it wrote, not proof that the client received it.

This is a local sidecar, not a hardened multi-tenant service. The route layer
does not implement API-key authentication or TLS. Keep loopback binding; any
remote exposure needs a separately configured trusted security boundary.
`POST /shutdown` is administrative and must not be exposed to untrusted clients.

See [the serving/API guide](lloom-engine/README.md) for completions, tokenization,
reasoning, history, decisions, images, batching, timing fields, and limitations.

## Configuration and memory

```sh
./target/release/lloom serve --help
./target/release/lloom options --compiled
./target/release/lloom options context
./target/release/lloom config --frontend serve --ctx-size 4096 --explain
./target/release/lloom options --schema
```

The generated tree is the option reference for **that binary**; the schema is
a projection, not a guarantee that every numeric/string spelling is equivalent.
Configuration precedence is defaults < environment < `--config FILE` < group
JSON < repeated `--set KEY=VALUE` < explicit flags; frontend defaults apply to
keys no layer set. Use `config --frontend serve` rather than assuming library,
chat, serve, and bench have identical defaults.

- Context includes prompt and generation. `--ctx-size` is a capacity, not a
  request to extend a checkpoint's context limit. The LFM2 GGUF loader refuses a
  capacity above its header limit; reduce the request or context on refusal.
- Prefill is tiled (`--prefill-tile`) to bound activation pressure; tile size,
  batch width, KV precision, resident weights and streamed slots all affect memory.
- The machine gate is on by default and combines host availability/reserve and
  recorded lloom reservations. Metal also has a working-set fraction budget
  (default `--memory-fraction 0.65`); these are admission checks, not OS-enforced
  quotas or a promise that an allocation/driver can never fail.
- Qwen4-exp expert streaming reduces the resident expert set, not the need for
  the checkpoint files or disk I/O. Qwen3.5 MoE currently loads experts resident.
- Do not use `--force` as an ordinary OOM remedy. Reduce context, batch, resident
  weights/slots or choose a smaller checkpoint. Avoid overlapping models on a
  constrained machine.

## A small local benchmark

Use the same release binary and checkpoint as serving, with explicit cells and
resolved defaults. This runs the existing in-process benchmark, not a new
model download or a Pipette orchestration layer:

```sh
./target/release/lloom config --frontend bench --ctx-size 4096 --explain
./target/release/lloom bench --model /path/to/LFM2.5-2.6B --ctx-size 4096 \
  --mode cells --cells prefill:128,decode:128:64 --runs 3
```

Read the per-run `BENCH` lines and their bracket/actual device. The first run
is cold; later repetitions do not erase page-in/compilation costs from the
first. Keep other inference/build work off the machine for timing comparisons;
record the binary, backend, checkpoint, cells and resolved configuration. This
is not a claim that a three-run example establishes a new release benchmark.
External Pipette clients can instead use the existing HTTP benchmark endpoints.

## Benchmarks

Measured, with the harness, prompts and limitations, in
[docs/benchmark.md](docs/benchmark.md). The cross-engine comparison there
(LFM2.5-2.6B on one M1 Max, 2026-10-21, cooled one-shot cells: lloom vs
llama.cpp vs MLX, one client driving each engine's own server) reads, in
short: one stream without speculation lloom decodes fastest (62–64 tok/s
against 60 and 52) and prefills slowest at long prompts (~2370 tokens: 1.68 s
against 0.96 s and 1.17 s); batched without speculation it leads at B=16/32 on
short prompts (616 vs 363 tok/s at B=32) and is level at ~2370 tokens; with
DSpark it is level with llama.cpp single-stream and its batched decode holds
up where llama.cpp's does not. The known gaps — long-prompt and batched
prefill, long-context batched decode, the idle-GPU ramp every engine pays —
are listed there with what is being done.

## Optional optimized diagnostic build

Keep diagnostic runs separate from the shipping binary and check the
capabilities of the actual build:

```sh
cargo build --profile release-with-debug -p lloom-engine --bin lloom \
  --features metal,serve,profile
./target/release-with-debug/lloom options --compiled
./target/release-with-debug/lloom chat --model /path/to/LFM2.5-2.6B \
  --ctx-size 4096 -v -p 'Say hello.'
```

Plain release is optimized, aborts on panic and strips symbols.
`release-with-debug` retains symbols and unwinding; `profile` enables profiling
instrumentation. Do not infer complete recorder compile-out simply from a
release build: that finer-grained diagnostics cleanup is separate work, not a
guarantee of this snapshot. Timing, tracing and symbolized investigation are
distinct from throughput measurements. Missing telemetry is not a genuine zero;
mandatory accounting and safety checks are not optional profiling features.
See [the build manifest](Cargo.toml) and [engine features](lloom-engine/Cargo.toml).

## Correctness, performance and scope

The reproducibility contract is for a fixed build, input, seed and supported
execution configuration. Different dtypes, kernels, math modes or models can
legitimately change floating-point results and tokens; digest identity alone
cannot establish useful model output. The project uses numerical references,
causal mutation checks, repeatability and real text/model gates together.

Historical LFM2.5-2.6B/M1 Max measurements are documented with their exact
source, prompts and configuration in the full repository's `design/measurements/`.
They are not current throughput promises for CPU, GGUF, other models or hardware.
The cross-engine tables in [docs/benchmark.md](docs/benchmark.md) cover
LFM2.5-2.6B on one M1 Max only.

## Branches and public history

This public tree is the `candle/` subtree of a larger development repository
that also holds the design notes, the issue tracker and the measurements. It is
published with `git subtree split` onto the fork's own history, so the history
here is upstream Candle followed by the fork's commits, unrewritten. `#NNN`
references in commit messages name issues of that development tracker, not
this repository's. Pre-release snapshots are published from the development
line (`dev`) as they pass its gates; this one is the snapshot tagged `pre-release-20261020` in the development
repository.

## Project and provenance

- `lloom-tensor`: tensor kernels and CPU/Metal/CUDA backends.
- `lloom`: model and sequence/decode primitives.
- `lloom-engine`: CLI, request frontend and HTTP serving.
- In the full repository, `design/` holds the specification and evidence;
  `amber/` is the consuming local voice assistant, not the serving CLI.

Forked from Candle; licensed under [MIT](LICENSE-MIT) or
[Apache-2.0](LICENSE-APACHE). Model weights/tokenizers have their own licenses.

For contributors: the design index (`DESIGN.md`), the development protocol
(`CONTRIBUTING.md`), the evidence tree (`measurements/`) and the architecture
documentation live in the development repository's `design/`; none is an
installation prerequisite for this tree.
