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
#
# `LLOOM_FUZZ_ASAN=1` adds `-Zsanitizer=address` to the SAME crates (nightly
# rustc): the sanitizer on the engine and the harness, not on LibAFL and its
# ~200 dependencies -- a global `-Zsanitizer=address` does not link on
# aarch64-apple-darwin (`ld: initializer pointer has no target` in LibAFL's
# `ctor` dependency). rustc links the ASan runtime when the final crate is
# sanitized, which every `fuzz_*` binary is. `--cfg lloom_fuzz_asan` compiles
# the arm's own positive controls (`record::sanitizer_controls`).
rustc="$1"; shift
extra=""
if [ "${LLOOM_FUZZ_ASAN:-0}" = "1" ]; then
  extra="-Zsanitizer=address --cfg lloom_fuzz_asan"
fi
case " $* " in
  *" --crate-name lloom "*|*" --crate-name lloom_tensor "*|*" --crate-name lloom_fuzz "*|*" --crate-name fuzz_"*)
    exec "$rustc" "$@" \
      -Cpasses=sancov-module \
      -Cllvm-args=-sanitizer-coverage-level=3 \
      -Cllvm-args=-sanitizer-coverage-trace-pc-guard \
      -Cllvm-args=-sanitizer-coverage-trace-compares \
      $extra
    ;;
  *)
    exec "$rustc" "$@"
    ;;
esac
