#!/bin/sh
# The rustc wrapper that puts SanitizerCoverage on the ENGINE crates and the
# harness, and on nothing else (lloom #706).
#
# rustc exposes no allowlist (`-fsanitize-coverage-allowlist` is a clang driver
# option, rejected as an LLVM arg) and per-package rustflags are nightly-only,
# so a wrapper is the one stable way to instrument three crates out of ~200.
# Measured without it: 1 181 290 guards, a 2 MiB map scanned per execution,
# ~690 exec/s on a host-only model. Deterministic: the flags depend only on
# `--crate-name`, so cargo's fingerprints stay honest.
#
# Wired by `.cargo/config.toml` (`build.rustc-wrapper`). Build scripts and
# proc-macros pass through untouched (their crate names never match).
rustc="$1"; shift
case " $* " in
  *" --crate-name lloom "*|*" --crate-name lloom_tensor "*|*" --crate-name lloom_fuzz "*|*" --crate-name fuzz_"*)
    exec "$rustc" "$@" \
      -Cpasses=sancov-module \
      -Cllvm-args=-sanitizer-coverage-level=3 \
      -Cllvm-args=-sanitizer-coverage-trace-pc-guard \
      -Cllvm-args=-sanitizer-coverage-trace-compares
    ;;
  *)
    exec "$rustc" "$@"
    ;;
esac
