# The one image for development, the CPU test suites and CI, on amd64 and
# arm64 (lloom #939). A red CI run reproduces on any host that can run it:
#
#   docker build --target test -t lloom-ci:test -f candle/Dockerfile .   # the repo root
#   docker run --rm -v "$PWD:/work" -w /work/candle lloom-ci:test \
#       cpu-tests.sh lloom 290 token_ring_fuzz
#
# `.github/ci/in-container.sh` wraps the two commands with the cache mounts
# the workflows use. Stages:
#
#   dev      the pinned Rust toolchain, the build dependencies, cargo-nextest
#   test     `dev` plus the CI environment: no `.cargo/config.toml` (its
#            `target-cpu=native` is the dev machine's), the `ci` profile
#   cuda     NVIDIA's CUDA devel image plus the same toolchain: the CUDA
#            compile-check (`cargo check --features cuda`), no device needed
#   runtime  a placeholder for a CPU runtime image (nothing is published yet)
#
# aarch64: `gemm-f16` 0.19 assembles its `neon,fp16` micro-kernels whatever
# the target CPU, and the assembler refuses them without `fp16` enabled
# (`error: instruction requires: fullfp16`), so the arm64 build carries
# `-C target-feature=+fp16`. Every aarch64 CPU this project runs on (Apple
# silicon, Cortex-A76) has it; the flag changes no other codegen.
ARG RUST_VERSION=1.96.1
ARG NEXTEST_VERSION=0.9.148

FROM rust:${RUST_VERSION}-bookworm AS dev
ARG NEXTEST_VERSION
RUN apt-get update \
    && apt-get install -y --no-install-recommends pkg-config libssl-dev cmake clang python3 \
    && rm -rf /var/lib/apt/lists/*
# The architecture is read from the running stage (`uname -m`), not from
# BuildKit's TARGETARCH: the legacy builder (the NAS's docker) does not set it.
RUN set -eu; \
    case "$(uname -m)" in \
      x86_64) nextest=linux ;; \
      aarch64) nextest=linux-arm ;; \
      *) echo "no cargo-nextest build for $(uname -m)" >&2; exit 1 ;; \
    esac; \
    curl -LsSf "https://get.nexte.st/${NEXTEST_VERSION}/${nextest}" | tar zxf - -C /usr/local/cargo/bin; \
    cargo nextest --version
# The tree's `.cargo/config.toml` says `build.rustflags = target-cpu=native`
# (the dev machine's); a `target.<triple>.rustflags` in CARGO_HOME takes
# precedence over any `build.rustflags`, so the container builds the
# baseline x86-64 the CI runner builds, and arm64 adds the `fp16` gemm-f16
# needs (above). Verified in `design/measurements/issue-939.md` §5: the
# container's `lloom-tensor` SIMD paths are the scalar ones the runner takes.
RUN set -eu; \
    case "$(uname -m)" in \
      x86_64) printf '[target.x86_64-unknown-linux-gnu]\nrustflags = []\n' ;; \
      aarch64) printf '[target.aarch64-unknown-linux-gnu]\nrustflags = ["-C", "target-feature=+fp16"]\n' ;; \
    esac > /usr/local/cargo/config.toml; \
    cat /usr/local/cargo/config.toml
WORKDIR /work

FROM dev AS test
# The suites run from the checkout mounted at /work (`.github/ci/in-container.sh`);
# its gate scripts are on PATH from there, never copied into the image.
ENV CARGO_TERM_COLOR=never \
    CARGO_INCREMENTAL=0 \
    LLOOM_CI_IMAGE=1 \
    PATH=/work/.github/ci:${PATH}

FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04 AS cuda
ARG RUST_VERSION
ENV CUDA_COMPUTE_CAP=86 \
    RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:${PATH} \
    CARGO_TERM_COLOR=never
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates build-essential pkg-config libssl-dev git python3 \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path --profile minimal --default-toolchain "${RUST_VERSION}" \
    && rustc --version
ENV PATH=/work/.github/ci:${PATH}
WORKDIR /work

FROM debian:bookworm-slim AS runtime
# Placeholder: a CPU runtime image would copy a release `lloom` binary here.
# Nothing is published (lloom #939, out of scope), so this stage only
# records the shape.
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libssl3 && rm -rf /var/lib/apt/lists/*
WORKDIR /srv
