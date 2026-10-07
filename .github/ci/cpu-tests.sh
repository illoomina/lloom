#!/usr/bin/env bash
# The CPU test gate for one crate, with a non-vacuity floor.
#
#   cpu-tests.sh [--known-red <binary>::<test>]... <crate> <floor> [<test-binary> ...]
#
# Runs `cargo test -p <crate> --no-fail-fast` with no `metal` feature -- unit
# tests, the integration tests under tests/, doctests, and the examples' build.
# `--no-fail-fast` is what makes every test binary run: without it cargo stops
# at the first binary with a failure, so one red unit test hides every
# integration test behind it. The run is then judged on what it printed, not on
# cargo's exit status alone:
#
#   * every binary that started reported a `test result:` line. A binary that
#     crashed (a signal, an abort) prints none, and cargo's exit status cannot
#     tell that from an ordinary failing test.
#   * the set of failing tests is EXACTLY the --known-red set. A failure not in
#     the set fails the gate; so does a known red that passed or did not run,
#     because an entry left in the set after its test is fixed would hide that
#     test's next regression. A known red is named `<binary>::<test>`, where
#     <binary> is the path cargo prints after `Running` (`src/lib.rs`,
#     `tests/conv_tests.rs`) or `doc` for the doctests.
#   * the sum of `N passed` over every `test result:` line is >= <floor>. A
#     green run that executed zero tests is a gate that cannot catch a
#     regression, and cargo reports `ok`
#     for a binary whose every test is `#[cfg]`-gated away or filtered out.
#     The floor is a fraction of the count measured when it was set, so a
#     whole test binary silently vanishing trips it while ordinary churn does
#     not; the measured counts are in the workflow file.
#   * each named <test-binary> ran at least one test. The Metal-gated binaries
#     legitimately report `running 0 tests` here, so the total alone cannot see
#     a CPU fuzzer that stopped running -- the named check can.
#
# The verdict is keyed on the `running N tests` / `test result:` pair that the
# test harness prints, never on grepping for the word "error": cargo's own
# error channel carries the word on an ordinary test failure.
set -euo pipefail

known_reds=()
while [ "${1:-}" = "--known-red" ]; do
  if [ $# -lt 2 ]; then
    echo "--known-red needs a <binary>::<test> argument" >&2
    exit 2
  fi
  known_reds+=("$2")
  shift 2
done
if [ $# -lt 2 ]; then
  echo "usage: $0 [--known-red <binary>::<test>]... <crate> <floor> [<test-binary> ...]" >&2
  exit 2
fi
crate="$1"
floor="$2"
shift 2

# Run from the workspace root (two levels above this script) regardless of
# caller CWD.
cd "$(dirname "$0")/../.."

log="$(mktemp)"
known="$(mktemp)"
failed_set="$(mktemp)"
trap 'rm -f "$log" "$known" "$failed_set"' EXIT
if [ "${#known_reds[@]}" -gt 0 ]; then printf '%s\n' "${known_reds[@]}" | sort -u >| "$known"; fi

echo "=== cargo test -p $crate --no-fail-fast (no metal feature) ==="
# The log is parsed below, so cargo's own output is kept plain: the GitHub
# runner exports CARGO_TERM_COLOR=always, which wraps `Running` in ANSI escapes
# and made the first run of this gate report the fuzzer as never having run.
# The parse strips escapes as well, so a colouring cargo does not set is also
# harmless.
set +e
CARGO_TERM_COLOR=never cargo test -p "$crate" --no-fail-fast 2>&1 | tee "$log"
rc="${PIPESTATUS[0]}"
set -e
sed -i.bak $'s/\x1b\[[0-9;]*m//g' "$log" && rm -f "$log.bak"

# Per binary: its header, whether it reported a result, and each failing test
# qualified by the binary it ran in. A doctest's name carries spaces
# (`src/x.rs - f (line 3)`), so the name is everything between `test ` and
# ` ... FAILED`.
read -r headers results passed failed < <(awk -v failed_set="$failed_set" '
  $1 == "Running" { bin = ($2 == "unittests") ? $3 : $2; headers++; next }
  $1 == "Doc-tests" { bin = "doc"; headers++; next }
  /^test .* \.\.\. FAILED$/ {
    name = $0; sub(/^test /, "", name); sub(/ \.\.\. FAILED$/, "", name)
    print bin "::" name >> failed_set
  }
  /^test result:/ {
    results++
    for (i = 1; i <= NF; i++) {
      if ($(i + 1) == "passed;") passed += $i
      if ($(i + 1) == "failed;") failed += $i
    }
  }
  END { printf "%d %d %d %d\n", headers, results, passed, failed }
' "$log")
sort -u -o "$failed_set" "$failed_set"
listed="$(wc -l < "$failed_set" | tr -d ' ')"

echo
echo "=== verdict: $crate: cargo exited $rc; $headers binaries started, $results reported; $passed passed, $failed failed; ${#known_reds[@]} known red(s); floor $floor ==="
if [ "$results" -eq 0 ]; then
  echo "  [FAIL] no test binary reported a result -- nothing ran (a build failure?)"
  exit 1
fi
if [ "$headers" -ne "$results" ]; then
  echo "  [FAIL] $headers test binaries started but $results reported a result -- a binary"
  echo "         died without one (a crash or an abort is not a failing test)"
  exit 1
fi
if [ "$failed" -ne "$listed" ]; then
  echo "  [FAIL] the result lines count $failed failure(s) but $listed were parsed by name --"
  echo "         the harness output is not being read correctly"
  exit 1
fi
if [ "$rc" -eq 0 ] && [ "$failed" -ne 0 ]; then
  echo "  [FAIL] $failed test(s) failed but cargo exited 0 -- the harness output is not being read correctly"
  exit 1
fi
if [ "$rc" -ne 0 ] && [ "$failed" -eq 0 ]; then
  echo "  [FAIL] cargo exited $rc with no failing test -- a failure that is not a test result"
  exit "$rc"
fi
unexpected="$(comm -23 "$failed_set" "$known")"
stale="$(comm -13 "$failed_set" "$known")"
if [ -n "$unexpected" ]; then
  echo "  [FAIL] failing, and not a known red:"
  printf '%s\n' "$unexpected" | sed 's/^/         /'
fi
if [ -n "$stale" ]; then
  echo "  [FAIL] a known red that did not fail -- it passed or did not run; drop it from"
  echo "         the workflow's --known-red list (a stale entry hides its test's next regression):"
  printf '%s\n' "$stale" | sed 's/^/         /'
fi
if [ -n "$unexpected" ] || [ -n "$stale" ]; then
  exit 1
fi
if [ "$failed" -ne 0 ]; then
  echo "  [ok ] the $failed failing test(s) are exactly the known reds"
fi
if [ "$passed" -lt "$floor" ]; then
  echo "  [FAIL] only $passed tests passed; the floor is $floor. Either test binaries"
  echo "         vanished from the CPU build or the floor needs re-measuring."
  exit 1
fi
echo "  [ok ] $passed tests passed, every started binary reported"

# Each named binary must have run >= 1 test: the `running N tests` line that
# follows its `Running tests/<name>.rs (...)` header.
for bin in "$@"; do
  n="$(awk -v bin="$bin" '
    $1 == "Running" && $2 ~ ("^tests/" bin "\\.rs$") { want = 1; next }
    want && $1 == "running" { print $2; exit }
  ' "$log")"
  if [ -z "$n" ]; then
    echo "  [FAIL] test binary $bin did not run at all"
    exit 1
  fi
  if [ "$n" -lt 1 ]; then
    echo "  [FAIL] test binary $bin ran 0 tests -- it is gated or filtered off the CPU arm"
    exit 1
  fi
  echo "  [ok ] $bin ran $n test(s)"
done

echo
echo "CPU tests for $crate PASSED and were not vacuous."
