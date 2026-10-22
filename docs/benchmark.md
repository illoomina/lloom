# Benchmarks

What has been measured, how, and what it does not show. Numbers here are
reproducible measurements with their exact source and configuration; they are
not throughput promises for other models, hardware, backends or file formats.

**Status (2026-10-22, pre-release).** The cross-engine tables below are the
run of 2026-10-22 on the release's defaults (`dev` `3f6375fe2`'s engine), with
**every engine's prompt cache disabled and verified**: the earlier runs
(2026-10-19 and the first 2026-10-21 matrix) let llama.cpp's and MLX's host
prompt caches restore ~874 of the 2372-token prompt because the three prompt
lengths are nested and the per-request tag was per repetition, not per cell;
their long-prompt TTFT rows were too low by up to a third. In this run
llama.cpp runs with `--cache-reuse 0 --cache-ram 0 --no-cache-prompt` and
`cache_prompt: false` on every request, MLX with `--prompt-cache-size 0`, every
request carries its own nonce, and each server's own prompt-token count is
the check (llama.cpp `prompt eval … / 2383 tokens`, MLX `2385/2385`, lloom
`prefilled=2385` at the 2372-token cell). The earlier matrices stay in the full
repository for the record (`xbench-20261019/`, `xbench-20261021/`).

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

## Cross-engine comparison (LFM2.5-2.6B, M1 Max, 2026-10-22)

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
GPU that has been idle for 20 s, and on this machine an idle GPU takes ~70 ms
to come back up to speed (`lloom serve`, 100-token prompt, max_tokens 1:
137–147 ms after ≤ 2 s idle, 201–225 ms after ≥ 5 s idle; measured 2026-10-21,
`xbench-20261021/ab-lead/idle-gap.sh`). Every engine pays it in these tables.

Engines: lloom `dev` 3f6375fe2 with the chat/serve defaults (F16 weights);
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
| 1 | lloom | 233 ms | 681 ms | 1.69 s |
| 1 | llama.cpp | 136 ms | 577 ms | 1.45 s |
| 1 | MLX | 243 ms | 735 ms | 1.75 s |
| 4 | lloom | 455 ms | 2.11 s | 4.74 s |
| 4 | llama.cpp | 348 ms | 2.10 s | 5.13 s |
| 16 | lloom | 1.26 s | 7.59 s | 22.97 s |
| 16 | llama.cpp | 1.36 s | 8.40 s | 18.89 s |
| 32 | lloom | 2.08 s | 12.68 s | 34.42 s |
| 32 | llama.cpp | 2.65 s | 13.15 s | 32.98 s |

**With DSpark:**

| B | Engine | Prompt | ~100 | ~872 | ~2372 |
|---|---|---|---|---|---|
| 1 | lloom | prose | 230 ms | 712 ms | 1.68 s |
| 1 | lloom | code | 222 ms | 691 ms | 1.69 s |
| 1 | llama.cpp | prose | 151 ms | 608 ms | 1.52 s |
| 1 | llama.cpp | code | 135 ms | 612 ms | 1.51 s |
| 1 | MLX (mlx-dspark) | prose | 151 ms | 655 ms | 1.64 s |
| 1 | MLX (mlx-dspark) | code | 153 ms | 641 ms | 1.65 s |
| 4 | lloom | prose | 404 ms | 1.60 s | 4.12 s |
| 4 | lloom | code | 420 ms | 1.57 s | 4.09 s |
| 4 | llama.cpp | prose | 411 ms | 2.29 s | 5.36 s |
| 4 | llama.cpp | code | 350 ms | 2.24 s | 5.41 s |
| 4 | MLX (mlx-dspark) | prose | 6.12 s | 6.76 s | 12.12 s |
| 4 | MLX (mlx-dspark) | code | 4.10 s | 6.31 s | 11.15 s |
| 16 | lloom | prose | 1.07 s | 6.68 s | 18.20 s |
| 16 | lloom | code | 1.07 s | 6.45 s | 17.32 s |
| 16 | llama.cpp | prose | 1.59 s | 11.89 s | 23.71 s |
| 16 | llama.cpp | code | 1.66 s | 9.55 s | 21.23 s |
| 32 | lloom | prose | 1.92 s | 10.52 s | 30.66 s |
| 32 | lloom | code | 1.95 s | 10.50 s | 30.53 s |
| 32 | llama.cpp | prose | 3.23 s | 17.05 s | 41.96 s |
| 32 | llama.cpp | code | 3.25 s | 16.38 s | 41.97 s |

### Throughput

Decode tok/s summed over the B streams, each stream timed from its first to
its last generated token (prefill excluded; it shows in the TTFT tables).

**Without DSpark** (mean of the `prose` and `code` prompts):

| B | Engine | ~100 | ~872 | ~2372 |
|---|---|---|---|---|
| 1 | lloom | 64 | 63 | 62 |
| 1 | llama.cpp | 60 | 60 | 60 |
| 1 | MLX | 52 | 51 | 51 |
| 4 | lloom | 152 | 136 | 96 |
| 4 | llama.cpp | 184 | 187 | 131 |
| 16 | lloom | 450 | 248 | 128 |
| 16 | llama.cpp | 271 | 223 | 148 |
| 32 | lloom | 587 | 287 | 176 |
| 32 | llama.cpp | 358 | 225 | 143 |

**With DSpark** (the drafter's acceptance depends on the text, so the prompts
are kept apart):

| B | Engine | Prompt | ~100 | ~872 | ~2372 |
|---|---|---|---|---|---|
| 1 | lloom | prose | 114 | 98 | 87 |
| 1 | lloom | code | 216 | 124 | 112 |
| 1 | llama.cpp | prose | 106 | 115 | 94 |
| 1 | llama.cpp | code | 78 | 121 | 107 |
| 1 | MLX (mlx-dspark) | prose | 67 | 32 | 24 |
| 1 | MLX (mlx-dspark) | code | 42 | 58 | 41 |
| 4 | lloom | prose | 187 | 177 | 98 |
| 4 | lloom | code | 253 | 177 | 130 |
| 4 | llama.cpp | prose | 87 | 121 | 50 |
| 4 | llama.cpp | code | 118 | 113 | 72 |
| 4 | MLX (mlx-dspark) | prose | 138 | 144 | 98 |
| 4 | MLX (mlx-dspark) | code | 190 | 176 | 139 |
| 16 | lloom | prose | 311 | 196 | 107 |
| 16 | lloom | code | 376 | 209 | 126 |
| 16 | llama.cpp | prose | 60 | 53 | 40 |
| 16 | llama.cpp | code | 60 | 58 | 48 |
| 32 | lloom | prose | 395 | 220 | 113 |
| 32 | lloom | code | 433 | 235 | 132 |
| 32 | llama.cpp | prose | 64 | 55 | 37 |
| 32 | llama.cpp | code | 60 | 51 | 41 |

What this shows, and what it does not:

- One stream without speculation: lloom decodes fastest (62–64 tok/s against
  llama.cpp's 60 and MLX's 51–52) and its first token comes later: ~2370
  tokens 1.69 s against 1.45 s (llama.cpp) and 1.75 s (MLX); ~100 tokens
  233 ms against 136 ms and 243 ms. The 100-token row is dominated by what
  happens between lloom's in-process first token and its first streamed chunk
  (`lloom chat` gives 103 ms on the same prompt; the idle-GPU ramp above is
  part of it) — the serve path, not the model.
- Batched without speculation: lloom leads at B=16 and B=32 on short and
  mid prompts (587 vs 358 tok/s at B=32, ~100 tokens; 287 vs 225 at ~872)
  and at B=32 ~2370 tokens (176 vs 143); it trails at B=4 on every length
  (152 vs 184 at ~100) and at B=16 ~2370 (128 vs 148). TTFT under load is
  level or ahead at B=4 and B=32 and behind at B=16 ~2370 tokens (23.0 s vs
  18.9 s).
- Single-stream DSpark: lloom and llama.cpp are level (prose 114/98/87 vs
  106/115/94; code 216/124/112 vs 78/121/107 tok/s), and lloom no longer
  decays with context thanks to the in-place KV for the verify window. Raw
  mid-document continuations draft worse than chat answers do (a chat answer
  reaches ~190 tok/s at B=1).
- Batched DSpark: lloom's scales (B=16: 311–376 tok/s; B=32: 395–433) where
  llama.cpp's collapses (40–64), and its batched TTFT is lower at every B
  (B=32 ~2370: 30.6 s vs 42.0 s); `mlx-dspark` batches at B=4 in this build
  (98–190 tok/s) and its B=1 is far behind both.

## What changed since the earlier matrices

- **The comparators' prompt caches were on in the earlier runs** (above). With
  them off, llama.cpp's ~2370-token prefill is 1.45 s, not 0.96; MLX's 1.75,
  not 1.17; and llama.cpp's batched TTFT at B=16 ~2370 is 18.9 s, not 12.0.
  lloom's rows were never cache-assisted (`prefilled` = the whole prompt on
  every request). The published gaps of 2026-10-21 were therefore overstated:
  single-stream prefill is 1.17× behind llama.cpp (not 1.75×) and ahead of
  MLX; batched prefill is level to 1.2× behind, not 1.6×.
- **The "narrow-projection GEMM deficit" named on 2026-10-21 was an artifact**
  of the isolated benchmark (a new command queue per timed batch); in a bare
  harness lloom's GEMM kernel runs at llama.cpp's `mul_mm` rate on every
  prefill shape (`issue-1076-step0/`), so there is no GEMM kernel gap.
- The 2026-10-19 → 2026-10-21 cool-down change (sustained-load vs one-shot
  cells) still stands; the comparators gained more from it than lloom did.

## Known gaps and limitations

Stated so the tables are read correctly; each is tracked in the development
repository and none is hidden by the numbers above.

- **The first streamed token on `serve`.** ~100 ms between lloom's in-process
  first token and its first streamed chunk (`chat` 1560 vs `serve` 1668 ms at
  ~2370 tokens; 103 vs 233 ms at ~100), of which ~70 ms is the GPU's idle ramp
  that every engine pays after ≥ 5 s idle and the rest is the serve path. A
  keep-warm option and the serve-path cost are the open items (#1076 in the
  development tracker).
- **Long-prompt prefill.** 1.69 s at ~2370 tokens against llama.cpp's 1.45 s:
  the forward is ~81 % GEMM at ~98 % GPU busy with the GEMMs at llama.cpp's
  rate; the remaining difference is ~85 ms of non-GEMM forward time
  (elementwise, copies, norms — fusions are the lever) plus the serve path
  above. The tile loop, the page-pool gather and the host are not where the
  time goes (profiled 2026-10-20).
- **Batched decode at small B.** At B=4 lloom's plain decode trails llama.cpp
  (152 vs 184 tok/s at ~100 tokens) while leading at B ≥ 16; the ragged
  packing's fixed cost per step is the suspect. Open item.
- **B=16 at long context.** Prefill (23.0 vs 18.9 s) and decode (128 vs 148
  tok/s) trail at ~2370 tokens; prefilling pending prompts together does not
  help on this machine (the per-token prefill cost is flat in the batch; the
  growth is arrival order plus the GPU leaving boost under sustained load —
  `issue-1079-batched-prefill.md`). Open.
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

Harness, prompts and raw results: `design/measurements/xbench-20261022/` in
the full development repository (`xbench.py`, `matrix.sh`, `results.jsonl`,
`tables.py`, `provenance.txt`, every server's log with its prompt-token count);
the earlier runs in `xbench-20261019/` and `xbench-20261021/`.
