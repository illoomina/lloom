# Benchmarks

What has been measured, how, and what it does not show. Numbers here are
reproducible measurements with their exact source and configuration; they are
not throughput promises for other models, hardware, backends or file formats.

**Status (2026-10-24, pre-release).** The cross-engine tables below are the
run of 2026-10-24 on the release's defaults (`dev` `d738e8564`'s engine: the
GPU keep-warm and the fused SiLU·mul are on by default since 2026-10-23), with
**every engine's prompt cache disabled and verified**: llama.cpp runs with
`--cache-reuse 0 --cache-ram 0 --no-cache-prompt` and `cache_prompt: false` on
every request, MLX with `--prompt-cache-size 0`, every request carries its own
nonce, and each server's own prompt-token count is the check (llama.cpp
`prompt eval … / 2383 tokens`, MLX `2385/2385`, lloom `prefilled=2385` at the
2372-token cell). The runs before 2026-10-22 let the comparators' host prompt
caches restore part of the long prompts (their long-prompt TTFT rows were too
low by up to a third); the 2026-10-22 matrix was the first honest one and the
2026-10-24 run repeats it on the new defaults. The earlier matrices stay in
the full repository for the record (`xbench-20261019/`, `xbench-20261021/`,
`xbench-20261022/`).

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

## Cross-engine comparison (LFM2.5-2.6B, M1 Max, 2026-10-24)

One client drives each engine's own OpenAI-compatible server with the same
raw prompts over streaming `/v1/completions`: 128 generated tokens with EOS
ignored, temperature 0, prompts cut to exact token counts with the LFM2
tokenizer, a unique nonce on every request, and every engine's prompt cache
off (above). `prose` is Pipette's seed passage; `code` is the Python stdlib
`json` source.

**Method: one timed request per cell** after one untimed warmup, with a 20 s
cool-down before every timed request so the chip is back near its idle
baseline (fans on; the GPU cooled to ≤ 60 °C before the matrix). This is a
release-potential number, not a sustained-load one: back-to-back requests heat
the M1 Max within two or three cells and cost 5–8 % of decode and up to 30 % of
long-prompt TTFT. The cool-down also means every timed request arrives at a
GPU that has been idle for 20 s; on this machine macOS unwires the engine's
residency after ~10 s idle and an idle GPU takes ~70 ms to come back up to
speed (measured 2026-10-21, `xbench-20261021/ab-lead/idle-gap.sh`). lloom's
`serve` now keeps its residency set requested every second while idle, for
180 s after the last request (`server.keep_warm`, on by default on Metal at
B=1); llama.cpp does the same with its own residency heartbeat; MLX pays the
ramp in these tables.

Engines: lloom `dev` d738e8564 with the chat/serve defaults (F16 weights);
llama.cpp b11721 `llama-server` with `-fa on`, `-np B`, the F16 GGUF and its
prompt cache off; MLX via `uvx --from mlx-lm` (mlx-lm 0.32) `mlx_lm.server`
with the bf16 MLX build and `--prompt-cache-size 0`. DSpark uses LiquidAI's
`LFM2.5-2.6B-DSpark` drafter: lloom natively, llama.cpp with the F16 GGUF
sidecar (`--spec-type draft-dspark --spec-draft-n-max 10`), MLX through the
third-party `mlx-dspark` server (default settings, not tuned; its build
batches at B=4).

B is the number of concurrent requests (llama.cpp `-np B`, lloom
`--batch B`). MLX ran at B=1 only; `mlx-dspark` also ran at B=4.

### Latency: time to first token

Mean TTFT over the B requests, by prompt length in tokens.

**Without DSpark** (mean of the `prose` and `code` prompts):

| B | Engine | ~100 | ~872 | ~2372 |
|---|---|---|---|---|
| 1 | lloom | 121 ms | 591 ms | 1.56 s |
| 1 | llama.cpp | 135 ms | 580 ms | 1.45 s |
| 1 | MLX | 250 ms | 729 ms | 1.73 s |
| 4 | lloom | 456 ms | 2.09 s | 4.68 s |
| 4 | llama.cpp | 365 ms | 2.12 s | 5.11 s |
| 16 | lloom | 1.25 s | 7.20 s | 20.81 s |
| 16 | llama.cpp | 1.38 s | 7.75 s | 17.64 s |
| 32 | lloom | 1.99 s | 12.22 s | 33.65 s |
| 32 | llama.cpp | 2.66 s | 13.38 s | 33.65 s |

**With DSpark:**

| B | Engine | Prompt | ~100 | ~872 | ~2372 |
|---|---|---|---|---|---|
| 1 | lloom | prose | 127 ms | 601 ms | 1.58 s |
| 1 | lloom | code | 136 ms | 595 ms | 1.58 s |
| 1 | llama.cpp | prose | 152 ms | 610 ms | 1.52 s |
| 1 | llama.cpp | code | 155 ms | 617 ms | 1.48 s |
| 1 | MLX (mlx-dspark) | prose | 155 ms | 652 ms | 1.66 s |
| 1 | MLX (mlx-dspark) | code | 141 ms | 648 ms | 1.67 s |
| 4 | lloom | prose | 374 ms | 1.56 s | 4.02 s |
| 4 | lloom | code | 391 ms | 1.54 s | 4.05 s |
| 4 | llama.cpp | prose | 415 ms | 2.19 s | 5.39 s |
| 4 | llama.cpp | code | 447 ms | 2.32 s | 5.39 s |
| 4 | MLX (mlx-dspark) | prose | 4.10 s | 7.27 s | 11.80 s |
| 4 | MLX (mlx-dspark) | code | 5.77 s | 7.22 s | 10.69 s |
| 16 | lloom | prose | 997 ms | 5.95 s | 16.38 s |
| 16 | lloom | code | 1.00 s | 5.83 s | 15.99 s |
| 16 | llama.cpp | prose | 1.67 s | 11.15 s | 22.38 s |
| 16 | llama.cpp | code | 1.57 s | 9.03 s | 20.53 s |
| 32 | lloom | prose | 1.75 s | 10.63 s | 34.18 s |
| 32 | lloom | code | 1.79 s | 10.54 s | 31.08 s |
| 32 | llama.cpp | prose | 3.21 s | 16.70 s | 41.99 s |
| 32 | llama.cpp | code | 3.23 s | 16.41 s | 41.97 s |

### Throughput

Decode tok/s summed over the B streams, each stream timed from its first to
its last generated token (prefill excluded; it shows in the TTFT tables).

**Without DSpark** (mean of the `prose` and `code` prompts):

| B | Engine | ~100 | ~872 | ~2372 |
|---|---|---|---|---|
| 1 | lloom | 64 | 63 | 62 |
| 1 | llama.cpp | 60 | 60 | 60 |
| 1 | MLX | 51 | 51 | 51 |
| 4 | lloom | 151 | 138 | 98 |
| 4 | llama.cpp | 188 | 187 | 133 |
| 16 | lloom | 469 | 266 | 143 |
| 16 | llama.cpp | 297 | 238 | 148 |
| 32 | lloom | 654 | 288 | 175 |
| 32 | llama.cpp | 355 | 222 | 141 |

**With DSpark** — read these rows with the caveat below: each cell is one
request, and the drafter's acceptance moves with the text (the prompts are
kept apart for that reason):

| B | Engine | Prompt | ~100 | ~872 | ~2372 |
|---|---|---|---|---|---|
| 1 | lloom | prose | 63 | 63 | 78 |
| 1 | lloom | code | 136 | 93 | 107 |
| 1 | llama.cpp | prose | 98 | 73 | 74 |
| 1 | llama.cpp | code | 210 | 124 | 121 |
| 1 | MLX (mlx-dspark) | prose | 36 | 73 | 25 |
| 1 | MLX (mlx-dspark) | code | 45 | 57 | 52 |
| 4 | lloom | prose | 237 | 164 | 109 |
| 4 | lloom | code | 228 | 184 | 128 |
| 4 | llama.cpp | prose | 94 | 118 | 59 |
| 4 | llama.cpp | code | 118 | 92 | 75 |
| 4 | MLX (mlx-dspark) | prose | 197 | 165 | 103 |
| 4 | MLX (mlx-dspark) | code | 184 | 170 | 142 |
| 16 | lloom | prose | 377 | 216 | 112 |
| 16 | lloom | code | 428 | 228 | 135 |
| 16 | llama.cpp | prose | 63 | 69 | 41 |
| 16 | llama.cpp | code | 89 | 56 | 47 |
| 32 | lloom | prose | 428 | 224 | 107 |
| 32 | lloom | code | 438 | 235 | 130 |
| 32 | llama.cpp | prose | 59 | 55 | 38 |
| 32 | llama.cpp | code | 57 | 51 | 41 |

**The DSpark rows are single samples with a text-driven spread.** Every cell
is one request carrying its own nonce; under greedy decoding the continuation
differs with the nonce and the drafter's acceptance with it. On the same
B=1 ~100-token prose cell, five nonces gave lloom 63–150 tok/s — the verify
step rate is constant (~32 steps/s), the accepted tokens per step ranged
1.9–4.7. A same-prompt A/B (fixed nonce, three texts per cell, three
repetitions, `xbench-20261024/ab-b1spec/`) puts lloom's B=1 DSpark decode at
**69–83 tok/s (prose ~100), 93–98 (prose ~872), 125–135 (code ~100), 150–156
(code ~872)**, with the streamed text and the chunk count identical across the
binaries of 2026-10-22 and 2026-10-24 and with the fused SiLU·mul and the
keep-warm on or off. The DSpark cells are therefore not comparable across the
2026-10-22 and 2026-10-24 matrices, and the single-stream DSpark comparison
with llama.cpp is level within that spread. The plain-decode and TTFT rows
have no such spread (no drafter; the nonce changes the text, not the step
cost).

What this shows, and what it does not:

- One stream without speculation: lloom decodes fastest (62–64 tok/s against
  llama.cpp's 60 and MLX's 51) and its first token is now level: ~100 tokens
  121 ms against 135 ms (llama.cpp) and 250 ms (MLX); ~872 tokens 591 ms
  against 580 and 729; ~2370 tokens 1.56 s against 1.45 s and 1.73 s. The
  2026-10-22 gap at ~100 tokens (233 vs 136 ms) was the GPU's idle ramp
  after macOS unwired the residency set; the keep-warm closes it.
- Batched without speculation: lloom leads at B=16 and B=32 on short and
  mid prompts (654 vs 355 tok/s at B=32, ~100 tokens; 288 vs 222 at ~872)
  and at B=32 ~2370 tokens (175 vs 141); it trails at B=4 on every length
  (151 vs 188 at ~100) and is level at B=16 ~2370 (143 vs 148). TTFT under
  load is level or ahead at B=4 and B=32 and behind at B=16 ~2370 tokens
  (20.8 s vs 17.6 s).
- Single-stream DSpark: level with llama.cpp within the spread above; lloom
  does not decay with context (the in-place KV for the verify window). Raw
  mid-document continuations draft worse than chat answers do (a chat answer
  reaches ~190 tok/s at B=1).
- Batched DSpark: lloom's scales (B=16: 377–428 tok/s; B=32: 428–438) where
  llama.cpp's collapses (38–89), and its batched TTFT is lower at every B
  (B=32 ~2370: 31–34 s vs 42.0 s); `mlx-dspark` batches at B=4 in this build
  (103–197 tok/s) and its B=1 is far behind both.

## What changed since the earlier matrices

- **2026-10-24: the GPU keep-warm and the fused SiLU·mul are on by default.**
  Single-stream TTFT fell from 233 / 681 ms / 1.69 s to 121 / 591 ms / 1.56 s
  (~100 / ~872 / ~2370 tokens); plain decode is unchanged at B=1 and B=4 and
  4–12 % higher at B=16 and B=32 (the fusion). Streams are byte-identical to
  the 2026-10-22 engine's on the same prompts (`ab-b1spec/`).
- **The comparators' prompt caches were on in the runs before 2026-10-22.**
  With them off, llama.cpp's ~2370-token prefill is 1.45 s, not 0.96; MLX's
  1.73, not 1.17; and llama.cpp's batched TTFT at B=16 ~2370 is 17.6 s, not
  12.0. lloom's rows were never cache-assisted (`prefilled` = the whole prompt
  on every request).
- **The "narrow-projection GEMM deficit" named on 2026-10-21 was an artifact**
  of the isolated benchmark (a new command queue per timed batch); in a bare
  harness lloom's GEMM kernel runs at llama.cpp's `mul_mm` rate on every
  prefill shape (`issue-1076-step0/`), so there is no GEMM kernel gap.
- The 2026-10-19 → 2026-10-21 cool-down change (sustained-load vs one-shot
  cells) still stands; the comparators gained more from it than lloom did.

## Known gaps and limitations

Stated so the tables are read correctly; each is tracked in the development
repository and none is hidden by the numbers above.

- **Long-prompt prefill.** 1.56 s at ~2370 tokens against llama.cpp's 1.45 s:
  the forward is ~81 % GEMM at ~98 % GPU busy with the GEMMs at llama.cpp's
  rate; the remaining difference is non-GEMM forward time (elementwise,
  copies, norms — the SiLU·mul fusion took the first slice; more fusions are
  the lever). The tile loop, the page-pool gather and the host are not where
  the time goes (profiled 2026-10-20).
- **Batched decode at small B.** At B=4 lloom's plain decode trails llama.cpp
  (151 vs 188 tok/s at ~100 tokens) while leading at B ≥ 16; the batched
  step's fixed cost (packing, per-row readback, step composition) is the
  suspect. Open item, in work.
- **B=16 at long context.** Prefill trails at ~2370 tokens (20.8 vs 17.6 s);
  prefilling pending prompts together does not help on this machine (the
  per-token prefill cost is flat in the batch; the growth is arrival order
  plus the GPU leaving boost under sustained load —
  `issue-1079-batched-prefill.md`). Open.
- **DSpark acceptance is text-dependent** (above); the single-sample DSpark
  cells carry a ±2× spread at B=1 and are indicative, not a ranking.
- **Heat and boost.** The M1 Max leaves its boost clock ~6–7 s into
  sustained work (703 → 610–650 MHz, 44 → 29 W); long batched cells pay it
  on every engine. The one-shot method above avoids it; sustained-load
  numbers are lower.
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

Harness, prompts and raw results: `design/measurements/xbench-20261024/` in
the full development repository (`xbench.py`, `matrix.sh`, `results.jsonl`,
`tables.py`, `provenance.txt`, every server's log with its prompt-token count,
`ab-b1spec/` for the same-prompt A/B); the earlier runs in `xbench-20261019/`,
`xbench-20261021/` and `xbench-20261022/`.
