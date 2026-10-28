#!/usr/bin/env bash
# The CPU test gate for one crate, with a non-vacuity floor.
#
#   cpu-tests.sh [--known-red <binary>::<test>]... [--features <list>] <crate> <floor> [<module> ...]
#
# Runs the crate's tests with no `metal` feature under the `ci` profile
# (`[profile.ci]`, candle/Cargo.toml): `cargo nextest run` for the unit and
# integration tests -- one process per test, so a test that installs
# process-global state (a tuning group, an allocation counter, an env var)
# never runs under another's; `cargo test --doc` for the doctests, which
# nextest does not run, and that step builds the examples too. The run is
# judged on what it printed, not on an exit status alone:
#
#   * nextest's `Summary` line is present and its PASS/FAIL lines are parsed
#     by name. A binary that crashed is a FAIL line with its signal, never a
#     missing result.
#   * the set of failing tests is EXACTLY the --known-red set. A failure not
#     in the set fails the gate; so does a known red that passed or did not
#     run, because an entry left in the set after its test is fixed would
#     hide that test's next regression. A known red is named
#     `<binary>::<test>` with nextest's binary id: the crate's name for its
#     unit tests (`lloom-tensor::tuning::tests::x`), `<crate>::main` for the
#     integration binary (`lloom::main::turn_checkpoint::x`), `<crate>::<file>`
#     for a test file that owns its binary.
#   * the sum of passed tests (nextest + doctests) is >= <floor>. A green run
#     that executed zero tests is CONTRIBUTING.md §7.1's F-class failure (a
#     gate that cannot catch a regression). The floor is a fraction of the
#     count measured when it was set, so a whole test module silently
#     vanishing trips it while ordinary churn does not; the measured counts
#     are in the workflow file.
#   * each named <module> ran at least one test. The Metal-gated modules
#     legitimately contribute no test here, so the total alone cannot see a
#     CPU fuzzer that stopped running -- the named check can.
#   * a failing test is rerun, alone, under the `dev` profile with its
#     output shown: full debug info for the backtrace. The rerun's verdict
#     is printed; the gate's verdict is the `ci` run's.
#
# The verdict is keyed on nextest's PASS/FAIL/Summary lines and cargo's
# `test result:` lines, never on grepping for the word "error"
# (CONTRIBUTING.md §3.1b: cargo's own error channel carries the word on an
# ordinary failure).
set -euo pipefail

known_reds=()
features=()
while [ "${1:-}" = "--known-red" ] || [ "${1:-}" = "--features" ]; do
  if [ $# -lt 2 ]; then
    echo "$1 needs an argument" >&2
    exit 2
  fi
  case "$1" in
    --known-red) known_reds+=("$2") ;;
    --features) features=(--features "$2") ;;
  esac
  shift 2
done
if [ $# -lt 2 ]; then
  echo "usage: $0 [--known-red <binary>::<test>]... [--features <list>] <crate> <floor> [<module> ...]" >&2
  exit 2
fi
crate="$1"
floor="$2"
shift 2

# Run from the workspace root (two levels above this script) regardless of
# caller CWD.
cd "$(dirname "$0")/../.."

log="$(mktemp)"
doclog="$(mktemp)"
known="$(mktemp)"
failed_set="$(mktemp)"
failed_pairs="$(mktemp)"
trap 'rm -f "$log" "$doclog" "$known" "$failed_set" "$failed_pairs"' EXIT
if [ "${#known_reds[@]}" -gt 0 ]; then printf '%s\n' "${known_reds[@]}" | sort -u >| "$known"; fi

echo "=== cargo nextest run -p $crate --cargo-profile ci ${features[*]:-} (no metal feature) ==="
# The log is parsed below, so the output is kept plain: the GitHub runner
# exports CARGO_TERM_COLOR=always, which wraps the lines in ANSI escapes and
# made the first run of this gate's predecessor report the fuzzer as never
# having run. The parse strips escapes as well.
set +e
CARGO_TERM_COLOR=never cargo nextest run --locked -p "$crate" --cargo-profile ci --no-fail-fast \
  "${features[@]}" 2>&1 | tee "$log"
rc="${PIPESTATUS[0]}"
set -e
sed -i.bak $'s/\x1b\[[0-9;]*m//g' "$log" && rm -f "$log.bak"

echo
echo "=== cargo test --doc -p $crate --profile ci (the doctests; builds the examples) ==="
set +e
CARGO_TERM_COLOR=never cargo test --locked --doc -p "$crate" --profile ci "${features[@]}" 2>&1 | tee "$doclog"
docrc="${PIPESTATUS[0]}"
set -e
sed -i.bak $'s/\x1b\[[0-9;]*m//g' "$doclog" && rm -f "$doclog.bak"

# nextest: `PASS [ 0.014s] ( 1/45) lloom-engine decode::tests::x` and
# `FAIL [ 0.014s] ( 2/45) lloom::main turn_checkpoint::y` (a retried test
# prints `TRY n FAIL`; retries are off here), then the failures listed
# again after the summary, so a failing test is named twice and the set is
# deduplicated. The counts are the summary's:
# `Summary [ 0.098s] 45 tests run: 44 passed, 1 failed, 0 skipped`.
read -r started passed failed summary < <(awk -v failed_set="$failed_set" -v failed_pairs="$failed_pairs" '
  $1 == "Starting" { started = $2 }
  ($1 == "FAIL" && $2 == "[") || ($1 == "TRY" && $3 == "FAIL") {
    n = NF; bin = $(n - 1); name = $n
    print bin "::" name >> failed_set
    print bin "\t" name >> failed_pairs
  }
  $1 == "Summary" {
    summary = 1
    for (i = 1; i <= NF; i++) {
      if ($(i + 1) == "passed," || $(i + 1) == "passed") passed = $i
      if ($(i + 1) == "failed," || $(i + 1) == "failed") failed = $i
    }
  }
  END { printf "%d %d %d %d\n", started + 0, passed + 0, failed + 0, summary + 0 }
' "$log")
sort -u -o "$failed_set" "$failed_set"
sort -u -o "$failed_pairs" "$failed_pairs"
listed="$(wc -l < "$failed_set" | tr -d ' ')"

# doctests: cargo's `test result: ok. N passed; M failed; ...` lines.
read -r docresults docpassed docfailed < <(awk '
  /^test result:/ {
    results++
    for (i = 1; i <= NF; i++) {
      if ($(i + 1) == "passed;") passed += $i
      if ($(i + 1) == "failed;") failed += $i
    }
  }
  END { printf "%d %d %d\n", results + 0, passed + 0, failed + 0 }
' "$doclog")

total_passed=$((passed + docpassed))
echo
echo "=== verdict: $crate: nextest exited $rc ($started started; $passed passed, $failed failed); doctests exited $docrc ($docpassed passed, $docfailed failed); ${#known_reds[@]} known red(s); floor $floor ==="
if [ "$summary" -eq 0 ]; then
  echo "  [FAIL] nextest printed no Summary line -- nothing ran (a build failure?)"
  exit 1
fi
if [ "$docresults" -eq 0 ]; then
  echo "  [FAIL] the doctest step reported no result -- nothing ran (a build failure?)"
  exit 1
fi
if [ "$docfailed" -ne 0 ] || [ "$docrc" -ne 0 ]; then
  echo "  [FAIL] the doctests failed ($docfailed failed, cargo exited $docrc)"
  exit 1
fi
if [ "$failed" -ne "$listed" ]; then
  echo "  [FAIL] the summary counts $failed failure(s) but $listed were parsed by name -- the output is not being read correctly"
  exit 1
fi
if [ "$rc" -eq 0 ] && [ "$failed" -ne 0 ]; then
  echo "  [FAIL] $failed test(s) failed but nextest exited 0 -- the output is not being read correctly"
  exit 1
fi
if [ "$rc" -ne 0 ] && [ "$failed" -eq 0 ]; then
  echo "  [FAIL] nextest exited $rc with no failing test -- a failure that is not a test result"
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
if [ "$failed" -ne 0 ]; then
  # The rerun: each failing test alone, `dev` profile (full debug info),
  # output shown. Diagnostic; the verdict above stands.
  echo
  echo "=== rerun of the $failed failing test(s) under the dev profile, output shown ==="
  # The binary id and the test name both carry `::` (`lloom::main`,
  # `<file>::<test>`), so the pair is read tab-separated, as the parse kept it.
  while IFS=$'\t' read -r bin name; do
    echo "--- $bin $name"
    set +e
    CARGO_TERM_COLOR=never cargo nextest run --locked -p "$crate" --cargo-profile dev --no-capture \
      "${features[@]}" -E "binary_id($bin) & test(=$name)" 2>&1 \
      | sed $'s/\x1b\[[0-9;]*m//g' | grep -vE '^\s*(Compiling|Finished|Downloaded|Downloading)' | tail -60
    set -e
  done < "$failed_pairs"
fi
if [ -n "$unexpected" ] || [ -n "$stale" ]; then
  exit 1
fi
if [ "$failed" -ne 0 ]; then
  echo "  [ok ] the $failed failing test(s) are exactly the known reds"
fi
if [ "$total_passed" -lt "$floor" ]; then
  echo "  [FAIL] only $total_passed tests passed; the floor is $floor. Either test modules"
  echo "         vanished from the CPU build or the floor needs re-measuring."
  exit 1
fi
echo "  [ok ] $total_passed tests passed ($passed under nextest, $docpassed doctests)"

# Each named module must have run >= 1 test: a PASS/FAIL line whose test
# name starts with `<module>::`.
for module in "$@"; do
  n="$(awk -v mod="$module" '
    ($1 == "PASS" || $1 == "FAIL") && index($NF, mod "::") == 1 { n++ }
    END { print n + 0 }
  ' "$log")"
  if [ "$n" -lt 1 ]; then
    echo "  [FAIL] module $module ran 0 tests -- it is gated or filtered off the CPU arm, or no longer a module of tests/main.rs"
    exit 1
  fi
  echo "  [ok ] $module ran $n test(s)"
done

echo
echo "CPU tests for $crate PASSED and were not vacuous."
