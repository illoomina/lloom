#!/usr/bin/env bash
# CUDA compile-check positive control.
#
# A green CI that cannot catch a broken backend is the failure this job exists
# to prevent. So the job does not merely assert the tree compiles under
# `--features cuda`; it shows the check FAILING on a deliberately broken CUDA
# kernel, then passing again once restored.
#
# The discipline, applied to a CI gate:
#   rule 1 — run the non-vacuity (baseline) arm FIRST. A green baseline proves
#            the CUDA toolkit is present and the tree compiles; without it, every
#            RED below is uninterpretable (a missing nvcc fails the same way).
#   rule 2 — mutate IN PLACE, restore, and verify the restore by SHA-256.
#   rule 3 — key the verdict on what the compile RESOLVES to (cargo's exit code),
#            never on grepping the compiler's output for the word "error" — cargo
#            prints "error:" on ordinary failures too.
#
# Compile only, never run: `cargo check` compiles the .cu kernels to PTX (nvcc,
# via candle-kernels/build.rs) and type-checks the cudarc bindings. No kernel is
# launched and no GPU is required.
set -euo pipefail

# Run from the workspace root (two levels above this script) regardless of
# caller CWD.
cd "$(dirname "$0")/../.."

# lloom-tensor carries the `cuda` feature (`cuda = ["cudarc",
# "dep:candle-kernels", ...]`) and is the crate whose PTX build + cudarc
# bindings this gate compile-checks.
#
# `--lib --tests`: the test targets carry `#[cfg(feature = "cuda")]` code that
# no other gate compiles — the CPU job builds them without `cuda`, the Metal
# suite without `cuda` — so a lib-only check leaves them unchecked everywhere.
# Every arm below runs the same CHECK, so the baseline arm is the whole gate.
CHECK=(cargo check --locked -p lloom-tensor --features cuda --lib --tests)

# A stable, non-excluded kernel in the PTX build set (candle-kernels/build.rs
# scans src/*.cu; affine.cu is not in its exclude list). The mutation is an
# appended invalid token, so it does not depend on the file's contents and
# fails nvcc deterministically.
#
# The rebuild is NOT vacuous under a warm build cache: cudaforge emits a
# `cargo:rerun-if-changed` for every scanned .cu (cudaforge builder.rs), so
# touching affine.cu re-runs nvcc even with Swatinem/rust-cache primed. A target
# outside that scanned set would be cached-over and the control would pass
# vacuously — keep the target inside src/*.cu.
TARGET=candle-kernels/src/affine.cu
BACKUP="$(mktemp)"

if [ ! -f "$TARGET" ]; then
  echo "FATAL: $TARGET not found — the positive control's target moved."
  echo "       Pick another non-excluded .cu (see candle-kernels/build.rs exclude list)."
  exit 2
fi

# sha256sum on Linux (the CUDA container), shasum -a 256 on macOS.
sha256() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  else shasum -a 256 "$1" | awk '{print $1}'; fi
}

before_sha="$(sha256 "$TARGET")"
cp -f "$TARGET" "$BACKUP"
restore() { cp -f "$BACKUP" "$TARGET"; rm -f "$BACKUP"; }
trap restore EXIT INT TERM

echo "=== BASELINE: the unmutated tree must compile under --features cuda ==="
if "${CHECK[@]}"; then
  echo "  [ok ] baseline compiles — toolkit present, tree clean; RED below is now interpretable"
else
  echo "  [FAIL] baseline does not compile — nothing below is interpretable."
  echo "         Either the CUDA toolkit is missing or the fork's CUDA path is broken."
  exit 1
fi

echo
echo "=== MUTATION: break the CUDA kernel $TARGET ==="
printf '\n#error "CI positive control: deliberately broken CUDA kernel"\n' >> "$TARGET"
echo "  appended an #error directive; nvcc must now reject this translation unit"

echo
echo "=== THE MUTATED TREE MUST FAIL ==="
if "${CHECK[@]}"; then
  echo "  [FAIL] the broken CUDA kernel still compiles — the gate is blind to a"
  echo "         backend regression, so every green run it reports is unsupported."
  exit 1
else
  echo "  [ok ] the broken CUDA kernel is rejected — the gate can see a regression"
fi

echo
echo "=== RESTORE: and it must compile again ==="
restore
trap - EXIT INT TERM
after_sha="$(sha256 "$TARGET")"
if [ "$before_sha" != "$after_sha" ]; then
  echo "  [FAIL] restore did not reproduce the original file"
  echo "         before=$before_sha after=$after_sha"
  exit 1
fi
if "${CHECK[@]}"; then
  echo "  [ok ] restored tree compiles — the failure was the mutation, not drift"
else
  echo "  [FAIL] restored tree does not compile"
  exit 1
fi

echo
echo "Positive control PASSED: the CUDA compile-check detects a broken kernel."
