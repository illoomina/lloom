# lloom-wasm — browser-runner feasibility spike (#656)

Proves the engine RUNS in a browser at all: `lloom`'s CPU backend compiled to
`wasm32-unknown-unknown` (no metal), decoding in a Web Worker and producing the
**byte-identical CPU-reference result** (the §2.3.5a oracle).

The deliverable is the finding (does it run + the numbers), recorded in the PR
for #656. This crate is the harness that produced it.

## The answer: yes

- `lloom` + `lloom-tensor` compile to `wasm32-unknown-unknown` with the CPU
  backend and **no metal** (`dummy_metal_backend` stub only — verified from the
  artifact, `build_info().lloom_metal_feature == false`).
- A fixed matmul+softmax runs in a **Web Worker** and its f32-bit fingerprint
  **matches the native CPU reference exactly** (`c088a7abff0884d0`).
- Browser-ready wasm size (post `wasm-bindgen`, pre `wasm-opt`): **~546 KB**.

Three source changes to the engine were needed to make it compile to wasm; all
are additive and cfg-gated, so the native metal build is unchanged (see the PR).

## Reproduce

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.100   # must match the crate pin

# 1. native CPU reference (the oracle)
cargo run --release --bin reference          # prints fingerprint

# 2. wasm build + bindings
cargo build --release --target wasm32-unknown-unknown
wasm-bindgen --target web --out-dir www/pkg --out-name lloom_wasm \
    ../../lloom-wasm/target/wasm32-unknown-unknown/release/lloom_wasm.wasm
#   (or wherever CARGO_TARGET_DIR points)

# 3. serve + run in a browser Web Worker
cd www && python3 -m http.server 8656
#   open http://localhost:8656/ and click "Run smoke", OR drive headless:
node ../drive.mjs            # needs headless Chrome on :9876 (see drive.mjs)
```

The `.cargo/config.toml` enables `simd128` and selects getrandom's `wasm_js`
backend; both are required for the build.

## The fuller notebook: a real LFM2 token (#623)

`notebook.html` is a doc-style page with one runnable cell that decodes a real
LFM2 token in a Web Worker. It loads a **quantized LFM2.5-350M GGUF** (Q4_K_M,
~219 MB, fetched once from the Hub), reads the tokenizer embedded in the GGUF,
runs one forward pass through `quantized_lfm2::from_gguf` on the CPU backend, and
renders the token + top-k logit distribution in-cell.

Why quantized, not the raw checkpoint: a bf16 LFM2 is ~5 GB, over the wasm32
4 GiB linear-memory ceiling. The quantized GGUF is the real browser story — the
same path candle's own browser examples take.

```sh
# native oracle (the §2.3.5a reference): decode one token on the host CPU
curl -L -o /tmp/lfm2-350m-q4.gguf \
  https://huggingface.co/LiquidAI/LFM2.5-350M-GGUF/resolve/main/LFM2.5-350M-Q4_K_M.gguf
cargo run --release --bin reference_lfm2 -- /tmp/lfm2-350m-q4.gguf "The capital of France is"
#   -> token " Paris" (id 5242), logit fingerprint, top-8

# browser: build + bind (see above), then
ln -sf /tmp/lfm2-350m-q4.gguf www/LFM2.5-350M-Q4_K_M.gguf   # serve same-origin
cd www && python3 -m http.server 8656
#   open http://localhost:8656/notebook.html and click Run, OR headless:
GGUF=http://localhost:8656/LFM2.5-350M-Q4_K_M.gguf node ../drive_lfm2.mjs
```

Native and browser agree on the sampled token; the logit fingerprints differ
because the quantized dot product reduces in a different order on aarch64 NEON
than on wasm (same result, different arithmetic — §2.3.5a). Build with
`+simd128` (in `.cargo/config.toml`) for the wasm SIMD quantized kernels.

## What this is NOT

Not the multi-cell notebook FORMAT + server-side recompile runner (§7.6) — this
is the ONE end-to-end cell, the load-bearing proof. Not WebGPU/WGSL (the shader
cells are §623's separate option-C strand). Not a REPL. Not a throughput
benchmark — one 6-token prefill is a correctness proof, not a tok/s number.
