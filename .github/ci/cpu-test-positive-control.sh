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

# The token-ring tests are modules of lloom's one test binary (tests/main.rs);
# the `ci` profile is the gate's.
CHECK=(cargo nextest run --locked -p lloom --cargo-profile ci --no-fail-fast -E 'test(token_ring_fuzz::) | test(token_ring_tests::)')

TARGET=lloom/src/generation/token_ring.rs
ORIGINAL='.eq(&self.eos_ids)?'
MUTANT='.ne(&self.eos_ids)?'
# Keep custody receipts outside the source tree. Gate logs also carry complete
# base64 copies, since a CI container's temporary directory is not durable.
RECEIPTS="$(mktemp -d "${TMPDIR:-/tmp}/lloom-cpu-control.XXXXXX")"
BACKUP="$RECEIPTS/source.before"
REGRESSIONS=(lloom/proptest-regressions/token_ring_fuzz.txt lloom/tests/token_ring_fuzz.proptest-regressions)

if [ ! -f "$TARGET" ]; then
  echo "FATAL: $TARGET not found -- the positive control's target moved."
  exit 2
fi
if [ "$(grep -cF "$ORIGINAL" "$TARGET")" -ne 1 ]; then
  echo "FATAL: expected exactly one '$ORIGINAL' in $TARGET; the mutation site moved."
  exit 2
fi

sha256() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  else shasum -a 256 "$1" | awk '{print $1}'; fi
}

# token_ring_fuzz uses ProptestConfig::default(): SourceParallel persists beside
# the tests directory at proptest-regressions/token_ring_fuzz.txt. Keep the
# historical location too; either may already contain valid regression seeds.
for i in "${!REGRESSIONS[@]}"; do
  path="${REGRESSIONS[$i]}"
  if [ -L "$path" ] || { [ -e "$path" ] && [ ! -f "$path" ]; }; then
    echo "FATAL: refusing non-regular persistence path $path"
    exit 2
  fi
  if [ -f "$path" ]; then
    cp "$path" "$RECEIPTS/seed-$i.before"
    sha256 "$path" >"$RECEIPTS/seed-$i.before.sha256"
  fi
done
before_sha="$(sha256 "$TARGET")"
cp "$TARGET" "$BACKUP"

retain_seeds() {
  local phase="$1" i path
  for i in "${!REGRESSIONS[@]}"; do
    path="${REGRESSIONS[$i]}"
    if [ -f "$path" ]; then
      cp "$path" "$RECEIPTS/seed-$i.$phase" || return 1
      printf 'PERSISTENCE_RECEIPT phase=%s path=%s sha256=%s base64_begin\n' "$phase" "$path" "$(sha256 "$path")"
      base64 <"$path" || return 1
      printf '\nPERSISTENCE_RECEIPT base64_end\n'
    else
      printf 'PERSISTENCE_RECEIPT phase=%s path=%s absent\n' "$phase" "$path"
    fi
  done
}

restore() {
  cp "$BACKUP" "$TARGET" || return 1
  touch "$TARGET" || return 1
  [ "$(sha256 "$TARGET")" = "$before_sha" ] || return 1
  local i path
  for i in "${!REGRESSIONS[@]}"; do
    path="${REGRESSIONS[$i]}"
    if [ -f "$RECEIPTS/seed-$i.before" ]; then
      cp "$RECEIPTS/seed-$i.before" "$path" || return 1
      cmp "$RECEIPTS/seed-$i.before" "$path" || return 1
    elif [ -e "$path" ]; then
      rm -- "$path" || return 1
    fi
    if [ ! -f "$RECEIPTS/seed-$i.before" ] && [ -e "$path" ]; then return 1; fi
  done
  echo "PERSISTENCE_RESTORED source=$before_sha both_paths_verified=true receipts=$RECEIPTS"
}
active_child=
active_receipt=
group_quiescent() {
  [ -n "$active_receipt" ] || return 0
  python3 - "$active_receipt/exit.json" <<'GROUP'
import json,sys
r=json.load(open(sys.argv[1]));c=r.get('cleanup')or{}
assert r.get('reaped')and c.get('leader_reaped')and c.get('group_empty')and not c.get('errors'),r
GROUP
}
finish() {
  local rc=$?
  trap - EXIT INT TERM
  if [ -n "$active_child" ]; then
    kill -TERM "$active_child" 2>/dev/null || true
    wait "$active_child" || true
    active_child=
  fi
  if ! group_quiescent; then
    retain_seeds custody-failure || true
    echo "FATAL: group quiescence unverified; source/seeds retained, no racing restore"
    exit 125
  fi
  if ! retain_seeds exit; then rc=1; fi
  if ! restore; then echo "FATAL: restoration failed; custody retained at $RECEIPTS"; rc=1; fi
  exit "$rc"
}
trap finish EXIT
terminal_signal() {
  local sig="$1" code="$2"
  trap '' INT TERM
  if [ -n "$active_child" ]; then
    kill -"$sig" "$active_child" 2>/dev/null || true
    wait "$active_child" || true
    active_child=
  fi
  exit "$code"
}
trap 'terminal_signal INT 130' INT
trap 'terminal_signal TERM 143' TERM

check() {
  local phase="$1" rc
  active_receipt="$RECEIPTS/$phase-group"
  python3 .github/ci/cpu-command-group.py "$active_receipt" "${CHECK[@]}" >"$RECEIPTS/$phase.log" 2>&1 &
  active_child=$!
  if wait "$active_child"; then rc=0; else rc=$?; fi
  active_child=
  # Restore is forbidden until the retained group is reaped and empty.
  if ! group_quiescent; then exit 125; fi
  if ! python3 - "$active_receipt/exit.json" <<'WORKER'
import json,sys
r=json.load(open(sys.argv[1]))
assert not r.get('supervisor_error')and not r.get('timed_out')and not r.get('cancelled'),r
assert r['worker']['reaped']and r['worker']['error']is None,r
WORKER
  then echo "FATAL: command supervision failed, not an oracle verdict"; exit 125; fi
  cat "$active_receipt/stdout.txt" "$active_receipt/stderr.txt" >>"$RECEIPTS/$phase.log"
  printf 'COMMAND_GROUP_RECEIPT phase=%s\n' "$phase"
  cat "$active_receipt/exit.json" "$active_receipt/adopted-reaps.json"
  cat "$RECEIPTS/$phase.log"
  printf 'CONTROL_EXIT phase=%s exit=%s\n' "$phase" "$rc"
  return "$rc"
}

echo "=== BASELINE: the unmutated token-ring tests must pass ==="
if check baseline; then
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
if check mutant; then
  echo "  [FAIL] the inverted EOS flag passes the token-ring tests -- the gate is"
  echo "         blind to a regression in the mechanism it pins, so every green"
  echo "         run it reports is unsupported."
  exit 1
else
  # A compiler error or an empty/incorrect filter is not a killed mutation.
  if ! grep -Eq '^[[:space:]]*FAIL \[[^]]*\] \([0-9/ ]+\) [^ ]+ token_ring_(fuzz|tests)::' "$RECEIPTS/mutant.log" ||
     ! grep -Eq "(assertion .*failed|Test failed:|test failed:)" "$RECEIPTS/mutant.log"; then
    echo "  [FAIL] mutant did not produce a named runtime assertion failure"
    exit 1
  fi
  echo "  [ok ] the mutation is killed by a named runtime assertion"
fi
retain_seeds mutant

echo
echo "=== RESTORE: and it must pass again ==="
restore
if check restored; then
  echo "  [ok ] restored tree passes -- the failure was the mutation, not drift"
else
  echo "  [FAIL] restored tree fails"
  exit 1
fi

echo
echo "Positive control PASSED: the CPU test gate detects a broken TokenRing."
