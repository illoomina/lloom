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

## What this is NOT

Not the full notebook (#623), not WebGPU/WGSL, not a REPL. The smoke computation
is a fixed 6-element op — enough to prove the CPU kernels run and match the
oracle. A representative *token* latency needs the full LFM2 weights loaded from
JS, which is the next strand, not this spike (see the PR's "what was NOT done").
