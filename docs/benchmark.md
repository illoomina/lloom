# Benchmarks

What has been measured, how, and what it does not show. Numbers here are
reproducible measurements with their exact source and configuration; they are
not throughput promises for other models, hardware, backends or file formats.

**Status (2026-10-20, pre-release snapshot).** The cross-engine tables below are
the measurement of 2026-10-19 on `dev` `9bc89074a`. Several changes landed on
the engine after it and are **not** reflected in these numbers; the matrix is
scheduled to be re-run on the release's defaults, and this page will be updated
with the re-run, not retouched:

- the f16 verify-window rows (speculative decode draws its window through a
  multi-row f16 GEMV instead of the 16-row GEMM; DSpark on a prose prompt
  measured 77 → 96 tok/s in `lloom chat`, single runs, not cross-engine);
- the turn-reuse defaults on Metal (the EOS gate and a queued readback lead,
  so multi-turn requests reuse the cached conversation instead of
  re-prefilling it);
- history rendered from the model's own emitted ids, `reasoning_content` on
  the API, and a request whose sampler cannot ride the device readback ring
  taking the blocking arm with a warning instead of an error.

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

## Cross-engine comparison (LFM2.5-2.6B, M1 Max)

One client drives each engine's own OpenAI-compatible server with the same
raw prompts over streaming `/v1/completions`: 128 generated tokens with EOS
ignored, temperature 0, prompts cut to exact token counts with the LFM2
tokenizer, and a unique leading tag on every request so no engine can reuse a
cached prefix. Each cell is one warmup plus three reps, timed on the client.
`prose` is Pipette's seed passage; `code` is the Python stdlib `json` source.

Engines: lloom `dev` 9bc89074a with the chat/serve defaults (F16 weights);
llama.cpp b11721 `llama-server` with `-fa on`, `-np B` and the F16 GGUF;
MLX via mainline `mlx_lm.server` with the bf16 MLX build. DSpark uses
LiquidAI's `LFM2.5-2.6B-DSpark` drafter: lloom natively, llama.cpp with the
F16 GGUF sidecar (`--spec-type draft-dspark --spec-draft-n-max 10`), MLX
through the third-party `mlx-dspark` server (default settings, not tuned).

B is the number of concurrent requests (llama.cpp `-np B`, lloom
`--batch B`). MLX ran at B=1 only; `mlx-dspark` also ran at B=4.

### Latency: time to first token

Mean TTFT over the B requests, by prompt length in tokens.

**Without DSpark** (mean of the `prose` and `code` prompts):

| Engine | ~100 | ~870 | ~2370 |
|---|---|---|---|
| lloom | 124 ms | 765 ms | 2.00 s |
| llama.cpp | 165 ms | 816 ms | 1.55 s |
| MLX | 171 ms | 681 ms | 1.33 s |



| B | Engine | ~100 | ~870 | ~2370 |
| 4 | lloom | 451 ms | 2.71 s | 5.82 s |
| 4 | llama.cpp | 441 ms | 2.52 s | 5.03 s |
| 16 | lloom | 1.44 s | 7.47 s | 18.60 s |
| 16 | llama.cpp | 1.41 s | 7.65 s | 13.40 s |
| 32 | lloom | 2.21 s | 13.70 s | 35.80 s |
| 32 | llama.cpp | 2.86 s | 13.30 s | 23.60 s |

**With DSpark:**

| Engine | Prompt | ~100 | ~870 | ~2370 |
|---|---|---|---|---|
| lloom | prose | 117 ms | 764 ms | 2.05 s |
| lloom | code | 136 ms | 775 ms | 2.02 s |
| llama.cpp | prose | 127 ms | 812 ms | 1.45 s |
| llama.cpp | code | 162 ms | 819 ms | 1.44 s |
| MLX (mlx-dspark) | prose | 105 ms | 600 ms | 1.77 s |
| MLX (mlx-dspark) | code | 105 ms | 619 ms | 1.82 s |

| B | Engine | Prompt | ~100 | ~870 | ~2370 |
| 4 | lloom | prose | 313 ms | 1.78 s | 4.89 s |
| 4 | lloom | code | 333 ms | 1.83 s | 4.88 s |
| 4 | llama.cpp | prose | 495 ms | 2.75 s | 5.23 s |
| 4 | llama.cpp | code | 416 ms | 2.71 s | 4.99 s |
| 4 | MLX (mlx-dspark) | prose | 6.08 s | 7.90 s | 12.10 s |
| 4 | MLX (mlx-dspark) | code | 5.02 s | 4.97 s | 8.55 s |
| 16 | lloom | prose | 993 ms | 6.00 s | 16.20 s |
| 16 | lloom | code | 990 ms | 5.93 s | 16.20 s |
| 16 | llama.cpp | prose | 1.59 s | 8.60 s | 15.20 s |
| 16 | llama.cpp | code | 1.60 s | 8.63 s | 15.10 s |
| 32 | lloom | prose | 1.90 s | 11.50 s | 31.70 s |
| 32 | lloom | code | 1.87 s | 11.50 s | 31.70 s |
| 32 | llama.cpp | prose | 3.42 s | 16.50 s | 29.90 s |
| 32 | llama.cpp | code | 3.42 s | 16.40 s | 29.90 s |

### Throughput

Decode tok/s summed over the B streams, each stream timed from its first to
its last generated token (prefill excluded; it shows in the TTFT tables).

**Without DSpark** (mean of the `prose` and `code` prompts):

| B | Engine | ~100 | ~870 | ~2370 |
|---|---|---|---|---|
| 1 | lloom | 65 | 64 | 62 |
| 1 | llama.cpp | 47 | 50 | 51 |
| 1 | MLX | 52 | 52 | 51 |
| 4 | lloom | 128 | 120 | 89 |
| 4 | llama.cpp | 155 | 156 | 141 |
| 16 | lloom | 466 | 272 | 155 |
| 16 | llama.cpp | 299 | 230 | 178 |
| 32 | lloom | 512 | 283 | 175 |
| 32 | llama.cpp | 337 | 235 | 173 |

**With DSpark** (the drafter's acceptance depends on the text, so the prompts
are kept apart):

| B | Engine | Prompt | ~100 | ~870 | ~2370 |
|---|---|---|---|---|---|
| 1 | lloom | prose | 63 | 62 | 44 |
| 1 | lloom | code | 98 | 80 | 66 |
| 1 | llama.cpp | prose | 71 | 77 | 56 |
| 1 | llama.cpp | code | 103 | 99 | 88 |
| 1 | MLX (mlx-dspark) | prose | 33 | 31 | 24 |
| 1 | MLX (mlx-dspark) | code | 40 | 57 | 48 |
| 4 | lloom | prose | 183 | 143 | 100 |
| 4 | lloom | code | 223 | 166 | 114 |
| 4 | llama.cpp | prose | 78 | 77 | 61 |
| 4 | llama.cpp | code | 90 | 90 | 72 |
| 16 | lloom | prose | 320 | 200 | 118 |
| 16 | lloom | code | 385 | 222 | 128 |
| 16 | llama.cpp | prose | 62 | 57 | 44 |
| 16 | llama.cpp | code | 69 | 59 | 47 |
| 32 | lloom | prose | 352 | 209 | 118 |
| 32 | lloom | code | 411 | 228 | 129 |
| 32 | llama.cpp | prose | 54 | 54 | 41 |
| 32 | llama.cpp | code | 59 | 50 | 42 |

What this shows, and what it does not:

- One stream without speculation: lloom decodes fastest (62–65 tok/s
  against 47–52), but prefills more slowly at long prompts (~2370 tokens:
  TTFT 2.0 s against 1.5 s for llama.cpp and 1.3 s for MLX).
- Batched without speculation: lloom decodes fastest at B=16 and B=32 on
  short prompts (512 vs 337 tok/s at B=32), trails llama.cpp at B=4 (128 vs
  155), and is level with it at ~2370 tokens. Its slower prefill shows as
  higher TTFT under load (~2370 tokens, B=32: 35.8 s vs 23.6 s).
- DSpark depends on the text and the context length. These raw
  continuations of mid-document text draft worse than chat answers do (a chat
  answer reached 191 tok/s at B=1, 7.7 tokens per round). At B=1 llama.cpp's
  DSpark is ahead of lloom's and holds up better with length; lloom's is slower
  than plain decoding on prose at ~2370 tokens.
- Batched DSpark: lloom's pays at B=4 (183–223 vs 128 tok/s) and costs
  decode throughput at B=16 and B=32 on short prompts (385–411 vs 466–512);
  llama.cpp's serves batched requests far slower than its plain decode.
- `mlx-dspark` did not batch (at B=4 the requests ran one after another, so
  its B=4 row appears in the TTFT table only); its rows are not a verdict on
  speculation under MLX.

Harness, prompts and raw results: `design/measurements/xbench-20261019/` in the
full development repository (`xbench.py`, `matrix.sh`, `results.jsonl`, `tables.md`).


## Known gaps and limitations

Stated so the tables are read correctly; each is tracked in the development
repository and none is hidden by the numbers above.

- **Long-prompt prefill.** At ~2370 prompt tokens lloom's time to first token
  is 2.0 s against 1.55 s (llama.cpp) and 1.33 s (MLX), and the gap widens
  under batch (B=32: 35.8 s vs 23.6 s). Two causes are identified and in
  progress: on the paged KV layout every prefill tile after the first
  re-gathers and concatenates the accumulated K/V per attention layer (the
  same copy runs for the speculative verify window every round, which is why
  DSpark decays faster with context than plain decoding), and the admission
  estimate for prefill activations models buffers the fused attention kernel
  never allocates, so prefill tiles above 1024 tokens are refused. Both are
  pre-release work; the re-run will show what they moved.
- **Heat.** The 2026-10-19 matrix ran its three reps back to back. On the
  M1 Max the chip heats within two or three cells: the 2372-token TTFT is
  1.7 s from idle and 2.0–2.3 s sustained, decode 5–8 % lower. The harness now
  defaults to one timed request per cell after a cool-down (`--cooldown`,
  `--reps 1`); the re-run uses that shape. Treat the long-prompt TTFT rows as
  sustained-load numbers, not idle ones.
- **Long-context batched decode.** At ~2370 tokens and B=32, plain decode is
  level with llama.cpp (175 vs 173 tok/s) after leading by 50 % on short
  prompts; attention over long contexts under batch is where the lead goes.
  Not a pre-release item.
- **DSpark at B=1 on prose** trails llama.cpp's draft (63 vs 71 tok/s at
  ~100 tokens; 44 vs 56 at ~2370) and is slower than plain decoding at ~2370
  on prose. The drafter's acceptance depends on the text; chat answers draft
  far better than these raw mid-document continuations (191 tok/s at B=1 on a
  chat answer, 7.7 tokens per round — a single `lloom chat` run, not a matrix
  cell).
- **`mlx-dspark` did not batch** in this harness; its B=4 rows are not a
  verdict on speculation under MLX.
- **One model, one machine.** LFM2.5-2.6B (F16 safetensors) on one M1 Max,
  Metal. Qwen3.6 (dense), Qwen4-exp (MoE, resident and streamed experts) and
  the GGUF loaders run on the same engine, but have no cross-engine numbers
  here; internal single-run measurements exist in the development repository
  and are not reproduced on this page until they are run the same way. CPU
  execution and CUDA are not measured here at all.
- **Sampling.** Measurements are greedy (temperature 0). Filtered samplers
  (top-k, top-p) run on the host today and take the blocking readback arm; a
  device-side sampler is designed and not shipped. Their throughput is lower
  than the ring arm's and is not in these tables.
