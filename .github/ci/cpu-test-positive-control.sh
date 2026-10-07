#!/usr/bin/env bash
# CPU test-gate positive control.
#
# The CPU test job's other half (cpu-tests.sh) shows the suite RAN; this shows
# it can FAIL. A test that cannot fail is not a test, and a green gate that
# cannot see a regression is the failure the CI exists to prevent. So: break a production mechanism the CPU-runnable
# fuzzer pins, in place, and require the fuzzer to go red; then restore and
# require green again.
#
# The discipline is the same as cuda-positive-control.sh's:
#   rule 1 -- the non-vacuity (baseline) arm runs FIRST; without a green
#            baseline every RED below is uninterpretable.
#   rule 2 -- mutate IN PLACE, restore, and verify the restore by SHA-256.
#   rule 3 -- the verdict is cargo's exit code on the tests that RESOLVE the
#            mutation, never a grep of the output for "error".
#
# The target is `TokenRing::publish_step`'s EOS decision: the token is compared
# against the uploaded EOS ids with `eq`; flipping it to `ne` inverts the flag
# on every publish. The `token_ring_fuzz` model (proptest, CPU arm) and the
# `token_ring_tests` value tests both read the flag back, so the mutation is
# killed on a CPU-only host with no device involved.
set -euo pipefail

# Run from the workspace root (two levels above this script) regardless of
# caller CWD.
cd "$(dirname "$0")/../.."

CHECK=(cargo test -p lloom --test token_ring_fuzz --test token_ring_tests)

TARGET=lloom/src/generation/token_ring.rs
ORIGINAL='.eq(&self.eos_ids)?'
MUTANT='.ne(&self.eos_ids)?'
BACKUP="$(mktemp)"

if [ ! -f "$TARGET" ]; then
  echo "FATAL: $TARGET not found -- the positive control's target moved."
  exit 2
fi
if [ "$(grep -cF "$ORIGINAL" "$TARGET")" -ne 1 ]; then
  echo "FATAL: expected exactly one '$ORIGINAL' in $TARGET; the mutation site moved."
  echo "       Re-anchor the control on the EOS comparison in TokenRing::publish_step."
  exit 2
fi

# sha256sum on Linux, shasum -a 256 on macOS.
sha256() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  else shasum -a 256 "$1" | awk '{print $1}'; fi
}

# proptest persists the seeds of a failing case beside the test file. The
# mutated arm's failures are the control working, not regressions, so the
# persistence file is put back exactly as it was (absent, on this tree).
REGRESSIONS=lloom/tests/token_ring_fuzz.proptest-regressions
REGRESSIONS_BACKUP="$(mktemp)"
had_regressions=0
if [ -f "$REGRESSIONS" ]; then had_regressions=1; cp -f "$REGRESSIONS" "$REGRESSIONS_BACKUP"; fi

before_sha="$(sha256 "$TARGET")"
cp -f "$TARGET" "$BACKUP"
restore() {
  cp -f "$BACKUP" "$TARGET"; rm -f "$BACKUP"
  if [ "$had_regressions" -eq 1 ]; then cp -f "$REGRESSIONS_BACKUP" "$REGRESSIONS"; else rm -f "$REGRESSIONS"; fi
  rm -f "$REGRESSIONS_BACKUP"
}
trap restore EXIT INT TERM

echo "=== BASELINE: the unmutated token-ring tests must pass ==="
if "${CHECK[@]}"; then
  echo "  [ok ] baseline passes; RED below is now interpretable"
else
  echo "  [FAIL] baseline fails -- nothing below is interpretable."
  exit 1
fi

echo
echo "=== MUTATION: invert the EOS decision in $TARGET ==="
# python rather than sed: the pattern carries `&`, `?`, `.` and `(`, and a
# literal-string replace has no metacharacters to escape.
python3 - "$TARGET" "$ORIGINAL" "$MUTANT" <<'PY'
import sys
path, orig, mut = sys.argv[1:]
s = open(path).read()
assert s.count(orig) == 1
open(path, "w").write(s.replace(orig, mut))
PY
echo "  $ORIGINAL -> $MUTANT"

echo
echo "=== THE MUTATED TREE MUST FAIL ==="
if "${CHECK[@]}"; then
  echo "  [FAIL] the inverted EOS flag passes the token-ring tests -- the gate is"
  echo "         blind to a regression in the mechanism it pins, so every green"
  echo "         run it reports is unsupported."
  exit 1
else
  echo "  [ok ] the mutation is killed -- the gate can see a regression"
fi

echo
echo "=== RESTORE: and it must pass again ==="
restore
trap - EXIT INT TERM
after_sha="$(sha256 "$TARGET")"
if [ "$before_sha" != "$after_sha" ]; then
  echo "  [FAIL] restore did not reproduce the original file"
  echo "         before=$before_sha after=$after_sha"
  exit 1
fi
if [ "$had_regressions" -eq 0 ] && [ -f "$REGRESSIONS" ]; then
  echo "  [FAIL] the mutated arm left $REGRESSIONS behind"
  exit 1
fi
if "${CHECK[@]}"; then
  echo "  [ok ] restored tree passes -- the failure was the mutation, not drift"
else
  echo "  [FAIL] restored tree fails"
  exit 1
fi

echo
echo "Positive control PASSED: the CPU test gate detects a broken TokenRing."
