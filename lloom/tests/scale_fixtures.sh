#!/usr/bin/env bash
# The periodic real-scale run (docs/porting-models.md §21.8a): every test
# file that builds its model through `lloom::testing::fixture`, run with
# LLOOM_FIXTURE=<backend>, one binary at a time under `lloom-arb measure`
# at scale (a model there holds tens of GB of the one GPU), each test's FIXTURE
# line (wall time, device memory, peak) and the binary's verdict recorded.
#
#   lloom/tests/scale_fixtures.sh [real|synthetic|tiny] [out-dir]
#
# Run it before a release, and after any change to the allocator, the
# buffer pool, admission, a loader or the streamed expert tier: those are
# the mechanisms whose failures have appeared only at the real scale.
#
# Every arm runs whatever the others did, and the verdicts are collected
# and judged at the end. A binary that prints no FIXTURE line never reached
# the fixture, and one that ran no test ran nothing: both fail the run.
set -uo pipefail

backend=${1:-real}
out=${2:-scale-fixtures-$backend-$(date +%Y%m%dT%H%M%S)}
arb=${LLOOM_ARB:-lloom-arb}
here=$(cd "$(dirname "$0")" && pwd)
candle=$(cd "$here/../.." && pwd)
mkdir -p "$out"
out=$(cd "$out" && pwd)

cd "$candle"
files=$(grep -lE 'Fixture::(<|new)' lloom/tests/*.rs | xargs -n1 basename | sed 's/\.rs$//' | sort)

# Every file is a module of the one integration binary, `tests/main.rs`
# (CONTRIBUTING §3.1d): build it once and select each file's tests by its
# module path.
echo "== build (tests/main.rs) -> $out/build.json"
$arb share --wait-timeout 86400 -- \
    cargo test --release --features metal -p lloom --test main --no-run \
    --message-format=json >|"$out/build.json" 2>"$out/build.err"
rc=$?
if [ $rc -ne 0 ]; then
    tail -30 "$out/build.err"
    echo "BUILD FAILED rc=$rc"
    exit 1
fi

# A model at scale holds tens of GB of the one GPU: a lease of its own.
# The tiny models are a test run, and share.
mode=measure
[ "$backend" = tiny ] && mode=share

fail=0
printf '%-40s %-8s %-6s %-8s %s\n' binary verdict tests wall_s max_rss_gb >|"$out/summary.txt"
bin=$(python3 - "$out/build.json" main <<'EOF'
import json, sys
for line in open(sys.argv[1]):
    try:
        m = json.loads(line)
    except ValueError:
        continue
    t = m.get("target", {})
    if m.get("reason") == "compiler-artifact" and t.get("name") == sys.argv[2] \
            and "test" in t.get("kind", []) and m.get("executable"):
        print(m["executable"])
EOF
)
if [ -z "$bin" ]; then
    echo "no test binary built" | tee -a "$out/summary.txt"
    exit 1
fi
for f in $files; do
    log="$out/$f.txt"
    echo "== $f"
    $arb $mode --wait-timeout 86400 -- \
        /usr/bin/time -l env LLOOM_FIXTURE="$backend" "$bin" "$f::" --test-threads 1 --nocapture \
        >|"$log" 2>&1
    rc=$?
    ran=$(grep -Eo '^running [0-9]+ tests?' "$log" | awk '{s+=$2} END {print s+0}')
    result=$(grep -E '^test result:' "$log" | tail -1)
    # A FIXTURE line can share a line with the harness's `test … ...`.
    grep -o 'FIXTURE test=.*' "$log" >|"$log.fixtures"
    fixtures=$(wc -l <"$log.fixtures" | tr -d ' ')
    # A FIXTURE line on another backend is a switch that did not reach the
    # fixture: a run labelled real that ran the tiny model.
    other=$(grep -vc " backend=$backend " "$log.fixtures")
    wall=$(grep -Eo '[0-9.]+ real' "$log" | tail -1 | awk '{print $1}')
    rss=$(grep -E 'maximum resident set size' "$log" | awk '{printf "%.2f", $1/1e9}')
    verdict=ok
    if [ $rc -ne 0 ] || ! echo "$result" | grep -q ' ok\.'; then verdict=FAILED; fi
    if [ "$ran" -eq 0 ]; then verdict=RAN-NONE; fi
    if [ "$fixtures" -eq 0 ]; then verdict=NO-FIXTURE; fi
    if [ "$other" -ne 0 ]; then verdict=WRONG-BACKEND; fi
    [ "$verdict" = ok ] || fail=1
    printf '%-40s %-8s %-6s %-8s %s\n' "$f" "$verdict" "$ran" "${wall:-?}" "${rss:-?}" | tee -a "$out/summary.txt"
    sed 's/^/    /' "$log.fixtures" | tee -a "$out/summary.txt"
done
echo "backend=$backend out=$out" | tee -a "$out/summary.txt"
if [ $fail -ne 0 ]; then
    echo "SCALE FIXTURES FAILED" | tee -a "$out/summary.txt"
    exit 1
fi
echo "SCALE FIXTURES OK" | tee -a "$out/summary.txt"
