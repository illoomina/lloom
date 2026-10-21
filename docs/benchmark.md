# Benchmarks

What has been measured, how, and what it does not show. Numbers here are
reproducible measurements with their exact source and configuration; they are
not throughput promises for other models, hardware, backends or file formats.

**Correction pending (2026-10-21, evening).** Two findings since the tables
below were published, both being re-measured: (1) the per-request tag does
*not* defeat llama.cpp's prompt cache for these prompts — the three prompt
lengths are nested (the 2372-token prompt contains the 872-token one), and
llama.cpp's chunked cache reuse matched the shared text, so its TTFT rows are
too low; measured with its cache disabled, llama.cpp prefills the 2372-token
prompt in ~1.48 s (its own prompt-eval time 1.47 s), not 0.96 s, and MLX in
~1.75 s, against lloom's 1.68 s. (2) The "narrow-projection GEMM deficit" named
under *Known gaps* was an artifact of the isolated benchmark (a new command
queue per timed batch); in a bare harness lloom's GEMM kernel runs at
llama.cpp's `mul_mm` rate on every prefill shape. The remaining single-stream
prefill difference (~190 ms at 2372 tokens) is ~110 ms between the in-process
first token and the first streamed chunk on the serve path, plus ~85 ms of
non-GEMM forward time. The matrix is being re-run with every engine's prompt
cache disabled and a per-cell nonce; this page will be rewritten from that
run. Until then read the TTFT rows below as upper bounds for lloom and MLX and
as understated for llama.cpp.

**Status (2026-10-21, pre-release).** The cross-engine tables below are the
re-run of 2026-10-21 on the release's defaults (`dev` `3f6375fe2`), replacing
the 2026-10-19 matrix. Everything the pre-release landed is in them: the
one-attention-route and buffer-ownership checkpoints (#985), history rendered
from the model's emitted ids (decision 51), the host sampling contract, the
producer-driven commit for host/IO inputs (#944), the f16 verify-window rows
(#942), the turn-reuse defaults, the prefill activation estimate, the 64×32
GEMM tile for narrow prefill projections and the in-place KV for single-stream
DSpark (#1068). The 2026-10-19 numbers are kept in the full repository
(`design/measurements/xbench-20261019/`) for the record; the differences that
matter are listed under *What changed* below.

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

## Cross-engine comparison (LFM2.5-2.6B, M1 Max, 2026-10-21)

One client drives each engine's own OpenAI-compatible server with the same
raw prompts over streaming `/v1/completions`: 128 generated tokens with EOS
ignored, temperature 0, prompts cut to exact token counts with the LFM2
tokenizer, and a unique leading tag on every request so no engine can reuse a
cached prefix. `prose` is Pipette's seed passage; `code` is the Python stdlib
`json` source.

**Method: one timed request per cell** after one untimed warmup, with a 20 s
cool-down before every timed request so the chip is back near its idle
baseline (fans on; the GPU cooled to ≤ 60 °C before the matrix). This is a
release-potential number, not a sustained-load one: back-to-back requests heat
the M1 Max within two or three cells and cost 5–8 % of decode and up to 30 % of
long-prompt TTFT. Note what the cool-down also means for the TTFT rows: every
timed request arrives at a GPU that has been idle for 20 s, and on this
machine an idle GPU takes ~70 ms to come back up to speed (`lloom serve`,
100-token prompt, max_tokens 1: 137–147 ms after ≤ 2 s idle, 201–225 ms after
≥ 5 s idle; measured 2026-10-21, `xbench-20261021/ab-lead/idle-gap.sh`). Every
engine pays it in these tables; a warm first token is ~70 ms lower across the
board.

Engines: lloom `dev` 3f6375fe2 with the chat/serve defaults (F16 weights);
llama.cpp b11721 `llama-server` with `-fa on`, `-np B` and the F16 GGUF;
MLX via `uvx --from mlx-lm` (mlx-lm 0.32) `mlx_lm.server` with the bf16 MLX
build. DSpark uses LiquidAI's `LFM2.5-2.6B-DSpark` drafter: lloom natively,
llama.cpp with the F16 GGUF sidecar (`--spec-type draft-dspark
--spec-draft-n-max 10`), MLX through the third-party `mlx-dspark` server
(default settings, not tuned; the build `uvx` resolved on 2026-10-21 batches
at B=4, the 2026-10-19 one did not).

B is the number of concurrent requests (llama.cpp `-np B`, lloom
`--batch B`). MLX ran at B=1 only; `mlx-dspark` also ran at B=4.

### Latency: time to first token

Mean TTFT over the B requests, by prompt length in tokens.

**Without DSpark** (mean of the `prose` and `code` prompts):

| B | Engine | ~100 | ~872 | ~2372 |
|---|---|---|---|---|
| 1 | lloom | 205 ms | 695 ms | 1.68 s |
| 1 | llama.cpp | 143 ms | 530 ms | 957 ms |
| 1 | MLX | 243 ms | 699 ms | 1.17 s |
| 4 | lloom | 440 ms | 2.10 s | 4.74 s |
| 4 | llama.cpp | 313 ms | 1.84 s | 3.58 s |
| 16 | lloom | 1.27 s | 7.18 s | 20.51 s |
| 16 | llama.cpp | 1.06 s | 6.73 s | 11.98 s |
| 32 | lloom | 1.99 s | 12.55 s | 35.45 s |
| 32 | llama.cpp | 2.31 s | 11.59 s | 21.93 s |

**With DSpark:**

| B | Engine | Prompt | ~100 | ~872 | ~2372 |
|---|---|---|---|---|---|
| 1 | lloom | prose | 225 ms | 708 ms | 1.70 s |
| 1 | lloom | code | 221 ms | 699 ms | 1.67 s |
| 1 | llama.cpp | prose | 147 ms | 560 ms | 1.01 s |
| 1 | llama.cpp | code | 148 ms | 566 ms | 1.00 s |
| 1 | MLX (mlx-dspark) | prose | 178 ms | 677 ms | 1.64 s |
| 1 | MLX (mlx-dspark) | code | 158 ms | 652 ms | 1.64 s |
| 4 | lloom | prose | 426 ms | 1.60 s | 4.10 s |
| 4 | lloom | code | 408 ms | 1.60 s | 4.09 s |
| 4 | llama.cpp | prose | 331 ms | 1.92 s | 3.80 s |
| 4 | llama.cpp | code | 344 ms | 1.94 s | 3.77 s |
| 4 | MLX (mlx-dspark) | prose | 5.60 s | 7.73 s | 12.27 s |
| 4 | MLX (mlx-dspark) | code | 5.80 s | 5.22 s | 8.77 s |
| 16 | lloom | prose | 1.12 s | 6.01 s | 16.41 s |
| 16 | lloom | code | 1.09 s | 5.79 s | 15.86 s |
| 16 | llama.cpp | prose | 1.21 s | 9.89 s | 15.75 s |
| 16 | llama.cpp | code | 1.22 s | 7.87 s | 13.96 s |
| 32 | lloom | prose | 2.02 s | 10.51 s | 30.58 s |
| 32 | lloom | code | 2.02 s | 10.51 s | 30.57 s |
| 32 | llama.cpp | prose | 2.65 s | 14.61 s | 27.87 s |
| 32 | llama.cpp | code | 2.63 s | 14.57 s | 28.04 s |

### Throughput

Decode tok/s summed over the B streams, each stream timed from its first to
its last generated token (prefill excluded; it shows in the TTFT tables).

**Without DSpark** (mean of the `prose` and `code` prompts):

| B | Engine | ~100 | ~872 | ~2372 |
|---|---|---|---|---|
| 1 | lloom | 64 | 63 | 62 |
| 1 | llama.cpp | 60 | 60 | 60 |
| 1 | MLX | 52 | 52 | 51 |
| 4 | lloom | 152 | 137 | 98 |
| 4 | llama.cpp | 188 | 187 | 172 |
| 16 | lloom | 470 | 268 | 143 |
| 16 | llama.cpp | 301 | 242 | 182 |
| 32 | lloom | 616 | 290 | 175 |
| 32 | llama.cpp | 363 | 238 | 174 |

**With DSpark** (the drafter's acceptance depends on the text, so the prompts
are kept apart):

| B | Engine | Prompt | ~100 | ~872 | ~2372 |
|---|---|---|---|---|---|
| 1 | lloom | prose | 87 | 97 | 64 |
| 1 | lloom | code | 145 | 138 | 82 |
| 1 | llama.cpp | prose | 88 | 100 | 67 |
| 1 | llama.cpp | code | 126 | 142 | 87 |
| 1 | MLX (mlx-dspark) | prose | 36 | 30 | 24 |
| 1 | MLX (mlx-dspark) | code | 40 | 57 | 52 |
| 4 | lloom | prose | 186 | 147 | 106 |
| 4 | lloom | code | 246 | 189 | 113 |
| 4 | llama.cpp | prose | 79 | 80 | 63 |
| 4 | llama.cpp | code | 106 | 122 | 67 |
| 4 | MLX (mlx-dspark) | prose | 133 | 123 | 95 |
| 4 | MLX (mlx-dspark) | code | 163 | 213 | 178 |
| 16 | lloom | prose | 326 | 186 | 115 |
| 16 | lloom | code | 391 | 233 | 125 |
| 16 | llama.cpp | prose | 57 | 47 | 45 |
| 16 | llama.cpp | code | 80 | 70 | 44 |
| 32 | lloom | prose | 364 | 210 | 117 |
| 32 | lloom | code | 411 | 241 | 128 |
| 32 | llama.cpp | prose | 57 | 45 | 42 |
| 32 | llama.cpp | code | 66 | 56 | 40 |

What this shows, and what it does not:

- One stream without speculation: lloom decodes fastest (62–64 tok/s against
  llama.cpp's 60 and MLX's 52) and prefills slowest at long prompts: at ~2370
  tokens its TTFT is 1.68 s against 0.96 s (llama.cpp) and 1.17 s (MLX). At
  ~100 tokens the three engines are within the idle-GPU ramp described above
  (lloom 205 ms, llama.cpp 143 ms, MLX 243 ms).
- Batched without speculation: lloom leads at B=16 and B=32 on short prompts
  (616 vs 363 tok/s at B=32) and is level with llama.cpp at ~2370 tokens; it
  trails at B=4 (152 vs 188). Its slower batched prefill shows as higher TTFT
  under load (~2370 tokens, B=32: 35.5 s vs 21.9 s; B=16: 20.5 s vs 12.0 s).
- Single-stream DSpark: lloom and llama.cpp are now level (prose 87/97/64 vs
  88/100/67; code 145/138/82 vs 126/142/87); lloom's decay with context is
  gone with the in-place KV for the verify window (its ~2370-token prose cell
  went from 44 to 64 tok/s). Raw mid-document continuations draft worse than
  chat answers do (a chat answer reaches ~190 tok/s at B=1).
- Batched DSpark: lloom's scales (B=16: 326–391 tok/s; B=32: 364–411) where
  llama.cpp's collapses (45–80); `mlx-dspark` batches at B=4 in this build
  (123–213 tok/s) and its B=1 is far behind both.

## What changed since the 2026-10-19 matrix

- **The comparators were throttled on 2026-10-19** (three reps back to back):
  cooled, llama.cpp's single-stream decode went 47 → 60 tok/s and its
  ~2370-token prefill 1.55 → 0.96 s; MLX's prefill 1.33 → 1.17 s. lloom gained
  less from the cool-down (decode flat at 62–65; prefill 2.00 → 1.68 s), so
  the single-stream decode lead narrowed and the long-prompt prefill gap is
  wider than the old table showed.
- lloom's ~100-token TTFT reads 205 ms against 124 ms before: the old run's
  requests arrived at a warm GPU (no cool-down); the new one's at an idle GPU
  (the ~70 ms ramp above). Warm, `lloom serve`'s first token on that prompt is
  ~140 ms and `lloom chat`'s 103 ms.
- Single-stream DSpark at long context: +45 % (prose) / +24 % (code) at ~2370
  tokens from the in-place KV (decision 52); prefill −20 ms at ~2370 from the
  64×32 tile.

## Known gaps and limitations

Stated so the tables are read correctly; each is tracked in the development
repository and none is hidden by the numbers above.

- **Long-prompt prefill.** 1.68 s at ~2370 tokens against llama.cpp's 0.96 s
  and MLX's 1.17 s. The prefill is ~81 % GEMM at ~98 % GPU busy; the measured
  gap against MLX is in the narrow projections (`q`/`o`/`k`/`v` at m = 512:
  5.4–6.6 vs 8.8 TFLOP/s), and llama.cpp's own f16 GEMM path is faster still
  at this shape. A 32-wide tile or split-K kernel for those shapes is the
  post-release item (#1076 in the development tracker); the tile loop, the
  page-pool gather and the host are not where the time goes (profiled
  2026-10-20).
- **Batched prefill.** Prompts are prefilled in arrival order in 512-token
  pieces that ride the live rows' decode steps, so a later stream's TTFT grows
  by ~2 s per earlier prompt at ~2370 tokens; the mean TTFT is about half the
  last stream's. llama.cpp's batched prefill is ~1.6× faster. A scheduler
  policy item, post-release.
- **Long-context batched decode.** At ~2370 tokens and B=16/32, plain decode is
  level with llama.cpp (143–175 vs 174–182 tok/s) after leading by 1.5–1.7× on
  short prompts; attention over long contexts under batch is where the lead
  goes. Post-release.
- **The idle-GPU ramp** (~70 ms before the first token after ≥ 5 s idle)
  applies to every engine on this machine; a keep-warm option for `serve` is
  a possible follow-up, not a shipped feature.
- **One model, one machine.** LFM2.5-2.6B (F16 safetensors) on one M1 Max,
  Metal. Qwen3.6 (dense), Qwen4-exp (MoE, resident and streamed experts) and
  the GGUF loaders run on the same engine, but have no cross-engine numbers
  here; internal single-run measurements exist in the development repository
  and are not reproduced on this page until they are run the same way. CPU
  execution and CUDA are not measured here at all.
- **Sampling.** Measurements are greedy (temperature 0). Filtered samplers
  (top-k, top-p) run on the host today and take the blocking readback arm
  with a warning; a device-side sampler is designed and not shipped. Their
  throughput is lower than the ring arm's and is not in these tables.

Harness, prompts and raw results: `design/measurements/xbench-20261021/` in the
full development repository (`xbench.py`, `matrix.sh`, `results.jsonl`,
`tables.py`, `provenance.txt`, `ab-lead/` for the first-token experiments).
