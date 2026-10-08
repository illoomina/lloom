<div align="center">

<img src="assets/lloom-logo.jpg" alt="lloom" width="240" />

# lloom

**A specialized LLM inference engine for Apple silicon**

</div>

> **Preview release.** This is an early public cut of the engine. Interfaces,
> flags and file layouts will change without notice, and the sections marked
> *placeholder* below are written last, once the release's model list, option
> surface and measurements are final.

lloom is a hard fork of [candle](https://github.com/huggingface/candle) tuned
to run LLM decode at, or near, the memory-bandwidth ceiling on M-series GPUs,
with speculative decoding (DSpark, MTP) and paged KV for concurrent serving.

The engine is correctness-first: every optimization is either
**byte-identical** to the reference path or a **deliberate, documented,
toward-truth** change (see [Determinism](#determinism)).

---

## Supported models

> **Placeholder** — the supported-model list is written when the preview's
> model set is final.
<!-- preview placeholder: supported models (private tracker 932) -->

The reference model is LiquidAI's **LFM2.5-2.6B** with its DSpark drafter.

## Numbers

> **Placeholder** — cross-engine numbers (lloom, MLX, llama.cpp) on the
> preview's models are measured last, on the release tree, and land here with
> the commands that reproduce them.
<!-- preview placeholder: numbers (private tracker 326) -->

**Why decode has a ceiling.** A decode step is memory-bound: it streams the
weights once to produce one token, so `tok/s ≤ achieved bandwidth ÷ bytes read
per token`, and the weights dominate the bytes at short context. At B=1 the
only way past that ceiling is to get more than one token out of each weight
sweep — which is why the work beyond fast kernels is **speculation** (several
accepted tokens per sweep) and **batching** (several sequences per sweep).
Prefill is compute-bound and the ceiling does not apply to it.

---

## Quick start

Requires a recent Rust toolchain and macOS on Apple silicon (Metal). The
reference checkpoints (`LiquidAI/LFM2.5-2.6B` and `LiquidAI/LFM2.5-2.6B-DSpark`)
are read from the Hugging Face cache; point `--model` / `--draft-dir` at them
or let discovery find them.

```bash
cargo build --release --features metal,serve -p lloom-engine --bin lloom

# Greedy decode (the "crown" B=1 fast path):
./target/release/lloom chat -p "Explain how a hash map works."

# Speculative decode (DSpark) — faster, byte-identical output:
./target/release/lloom chat --dspark -v -p "Explain how a hash map works."
```

The `[turn N: …]` line on stderr reports the steady throughput (the wall minus
the prompt's submit) and the arm that ran; for DSpark the `[dspark: … tok/s
steady]` line (`-v`) is the decode rate. Without `-p`, `lloom chat` is a REPL
that extends its cache turn by turn.

> **Placeholder** — the option reference is written when the option surface
> is final. Until then `lloom knobs` lists every knob and `lloom <cmd> --help`
> every flag.
<!-- preview placeholder: flags (private tracker 933) -->

---

## Serving: `lloom serve`, and the pipette contract

`lloom serve` (and `lloom-serve`, the same server under the name LiquidAI's
[`pipette-clients`](https://github.com/Liquid4All/pipette-clients) runners
spawn) puts the engine on a loopback port speaking two contracts at once: the
OpenAI-compatible surface (`/v1/completions` with `ignore_eos` and an exact
`usage`, streaming `/v1/chat/completions` on the checkpoint's own chat
template, `/tokenize`, `/v1/models`) and the pipette benchmark sidecar
contract (`/prefill_throughput`, `/decode_throughput`, `/end_to_end_latency`,
`/max_memory_usage`, `/eval` + `/eval/abort`, a `{"kind":"ready"}` marker on
stdout).

```bash
cargo build --release --features metal,serve -p lloom-engine --bins
./target/release/lloom-serve --port 8080 &        # → {"kind":"ready","host":"127.0.0.1","port":8080}

curl -s localhost:8080/v1/completions -d '{"prompt":"Explain how a hash map works.","max_tokens":100,"ignore_eos":true}'
curl -s localhost:8080/decode_throughput -d '{"prompt_tokens":512,"decode_tokens":100}'
```

The endpoint table, what each timing brackets, and the flags are in
[`lloom-engine/README.md`](lloom-engine/README.md).

---

## Layout

| crate | what |
|---|---|
| `lloom-tensor` | the tensor layer and its backends: the Metal kernels, the CPU backend (the reference the kernels are tested against), CUDA |
| `lloom` | the inference core: blocks, models, the decode drivers, speculative decoding, paged KV |
| `lloom-engine` | the serving engine: the `DecodeSession` seam and the `lloom` / `lloom-serve` binaries |
| `lloom-macros` | the derive macros behind the option groups and the axis registry |

The `candle-*` directories are candle's own crates, kept for breadth; the CPU
and CUDA backends compile in CI.

---

## What makes it fast

- **Near-bandwidth B=1 decode** — precompiled pipelines, fused kernels
  (conv/rmsnorm/layernorm single-round, GEMV epilogues), device-resident
  sampling (argmax and gumbel-max), EOS off the hot path.
- **DSpark speculative decode** — a small LFM2 drafter proposes a block; the
  target verifies it in one forward and accepts the longest agreeing prefix, so
  output is **byte-identical to greedy** while amortizing the per-step weight
  sweep across accepted tokens.
- **Paged KV + concurrency** — a paged KV pool with per-sequence page tables and
  refcounted sharing, and inter-sequence pipelining that runs independent
  sequences' command buffers concurrently on one queue.

---

## Determinism

Correctness is a property of each optimization, not an afterthought. The subtle
part is floating point: **fast-math is on**, GEMV and GEMM accumulate in
different orders, and gumbel-max sampling injects RNG — so "is the output
*identical* to the reference?" needs a precise answer.

**The invariant held: a given build is bit-identical run to run.** Fixed-order
reductions, no float atomics, counter-based RNG — so the *same binary* produces
the *same tokens* every run, under any GPU load or batch composition. What is
*not* claimed is that two *different* builds match bit-for-bit: fast-math mode,
a different kernel or tile, or GEMV-vs-GEMM order can each differ by ~1 ULP, and
at a near-tie between two logits that can change a token.

**Two gates, composed, separate what a single check cannot:**

- **Digest gate** — N runs of a long multi-turn generation with **identical
  SHA-256** over the full token stream and the logits. This catches
  non-determinism and pins each build self-identical.
- **Text gate** — the output is still **coherent**, checked against fixed
  substrings. This catches corruption a digest cannot: a digest is equally
  stable whether the model answers correctly or emits identical garbage every
  run.

**Speculative decoding is exact by construction.** A greedy verify accepts a
draft token **iff it equals the target's argmax**, so a bad drafter costs
acceptance rate and **never** the output.

---

## Branches, history and pull requests

- **`dev`** is where work lands. **Open pull requests against `dev`.**
- **`main`** is the default branch and the stable line. It moves only by a
  merge from `dev`; nothing is committed to it directly.

GitHub proposes `main` as a new pull request's base, so pick `dev` when you
open one (`gh pr create --base dev`). A pull request opened against `main` is
retargeted to `dev` before review.

`dev` is published from the project's development repository, where the engine
lives beside its design notes and measurements, and a merged pull request is
merged back there. **`#NNN` references in commit messages refer to that
private tracker, not to this repository's issues.**

The history is candle's (upstream `huggingface/candle`), then the fork's own
commits on top of it, with build output that was once committed removed.

---

## Provenance and license

A hard fork of [candle](https://github.com/huggingface/candle), under the same
terms: dual-licensed MIT ([`LICENSE-MIT`](LICENSE-MIT)) or Apache-2.0
([`LICENSE-APACHE`](LICENSE-APACHE)). LFM2 / LFM2.5 and DSpark are LiquidAI's.
