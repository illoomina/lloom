#!/usr/bin/env bash
# Mutation harness for issue #749's mechanisms (CONTRIBUTING §3.1a form):
# the clean arm FIRST, mutate IN PLACE with perl, restore from a byte copy
# taken at start (never `git checkout` — the tree may be uncommitted),
# verify the restore by SHA-256, and key the verdict on cargo's
# `test result:` line rather than a grep for `error`.
#
#   bash lloom/tests/mutate_749_stream.sh            # CPU arms (M1, M3, M4, M5)
#   bash lloom/tests/mutate_749_stream.sh --metal    # + M2 (the Metal slot write), M6 (the snapshot alias)
#
# M1  Streamed::ensure maps an expert to the NEXT slot (a wrong-slot dispatch)
# M2  QMetalStorage::write_with lands every plane at offset 0 (Metal only)
# M3  Cache::rewind_window keeps the KV lengths (the KV half of the rollback)
# M4  the PLE row cache serves the neighbouring slot's bytes on a hit
# M5  accept_prefix accepts every draft (the verify no longer verifies)
# M6  RecurrentState snapshots with `Tensor::copy` — a shallow clone on Metal
#     (the defect #749 found on the real model; Metal only)
set -u
cd "$(dirname "$0")/../.."   # candle/
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$PWD/../target-mutate-749}"
METAL=0
[ "${1:-}" = "--metal" ] && METAL=1

ES=lloom/src/nn/expert_stream.rs
RS=lloom/src/nn/recurrent.rs
QM=lloom-tensor/src/quantized/metal.rs
MD=lloom/src/models/qwen4exp/model.rs
PL=lloom/src/models/qwen4exp/ple.rs
SP=lloom/src/nn/speculator.rs
BAK=$(mktemp -d)
for f in $ES $RS $QM $MD $PL $SP; do cp "$f" "$BAK/$(basename $f)"; done
sha() { shasum -a 256 "$1" | cut -d' ' -f1; }
S_ES=$(sha $ES); S_RS=$(sha $RS); S_QM=$(sha $QM); S_MD=$(sha $MD); S_PL=$(sha $PL); S_SP=$(sha $SP)
restore() {
  cp "$BAK/expert_stream.rs" $ES; cp "$BAK/recurrent.rs" $RS; cp "$BAK/metal.rs" $QM; cp "$BAK/model.rs" $MD
  cp "$BAK/ple.rs" $PL; cp "$BAK/speculator.rs" $SP
  [ "$(sha $ES)" = "$S_ES" ] && [ "$(sha $RS)" = "$S_RS" ] && [ "$(sha $QM)" = "$S_QM" ] && [ "$(sha $MD)" = "$S_MD" ] \
    && [ "$(sha $PL)" = "$S_PL" ] && [ "$(sha $SP)" = "$S_SP" ] \
    && echo "RESTORE ok (sha ${S_ES:0:8} ${S_RS:0:8} ${S_QM:0:8} ${S_MD:0:8} ${S_PL:0:8} ${S_SP:0:8})" \
    || { echo "RESTORE FAILED"; exit 2; }
}
trap restore EXIT

CPU_TESTS="--test qwen4exp_streamed_identity --test qwen4exp_mtp_identity"
# verdict: GREEN iff every `test result:` line reads `ok` and at least one test ran
run_cpu() {
  local out
  out=$(cargo test -p lloom $CPU_TESTS 2>&1)
  local results; results=$(echo "$out" | grep -c "^test result:")
  local passed; passed=$(echo "$out" | grep "^test result:" | grep -o "[0-9]* passed" | awk '{s+=$1} END {print s+0}')
  if echo "$out" | grep -q "^error\[E\|could not compile"; then echo "BUILD-ERROR"; return; fi
  if [ "$results" -gt 0 ] && [ "$passed" -gt 0 ] && ! echo "$out" | grep -q "^test result: FAILED"; then echo GREEN; else echo RED; fi
}
run_metal() {
  local out
  out=$(cargo test -p lloom --features metal --test qwen4exp_streamed_metal --test recurrent_state_metal 2>&1)
  local passed; passed=$(echo "$out" | grep "^test result:" | grep -o "[0-9]* passed" | awk '{s+=$1} END {print s+0}')
  if echo "$out" | grep -q "^error\[E\|could not compile"; then echo "BUILD-ERROR"; return; fi
  if [ "$passed" -gt 0 ] && ! echo "$out" | grep -q "^test result: FAILED"; then echo GREEN; else echo RED; fi
}

echo "== CLEAN (must be GREEN or nothing below is interpretable)"
clean=$(run_cpu); echo "CLEAN: $clean"
[ "$clean" = GREEN ] || { echo "clean arm not green; aborting"; exit 1; }
if [ $METAL = 1 ]; then cm=$(run_metal); echo "CLEAN metal: $cm"; [ "$cm" = GREEN ] || exit 1; fi

echo "== M1 wrong slot"
perl -0pi -e 's/slots\[i\] as u32\n/((slots[i] + 1) % self.n_slots) as u32\n/' $ES
grep -q "slots\[i\] + 1" $ES || echo "M1 did not apply"
m1=$(run_cpu); echo "MUTATED M1: $m1"; restore

if [ $METAL = 1 ]; then
  echo "== M2 metal slot write at offset 0"
  perl -0pi -e 's/(\n\s+0,\n\s+)offset,(\n\s+\)\n\s+\.map_err\(MetalError::from\)\?;\n\s+Ok\(\(\)\)\n\s+\}\n\n\s+pub fn data)/${1}0,$2/' $QM
  grep -c "            0,\n            0," $QM >/dev/null
  m2=$(run_metal); echo "MUTATED M2: $m2"; restore
fi

echo "== M3 rollback keeps the KV lengths"
perl -0pi -e 's/k\.truncate\(tokens, blocks\);/let _ = (tokens, blocks);/' $MD
grep -q "let _ = (tokens, blocks);" $MD || echo "M3 did not apply"
m3=$(run_cpu); echo "MUTATED M3: $m3"; restore

echo "== M4 the row cache serves the neighbouring slot on a hit"
perl -0pi -e 's/let start = s as usize \* self\.row_bytes;\n        Some\(/let start = ((s as usize + 1) % self.capacity) * self.row_bytes;\n        Some(/' $PL
grep -q "(s as usize + 1) % self.capacity" $PL || echo "M4 did not apply"
m4=$(run_cpu); echo "MUTATED M4: $m4"; restore

echo "== M5 accept everything"
perl -0pi -e 's/if eos_ids\.contains\(&target\) \|\| \(i < k && target != proposed\[i\]\) \{/if eos_ids.contains(&target) {/' $SP
grep -q "if eos_ids.contains(&target) {" $SP || echo "M5 did not apply"
m5=$(run_cpu); echo "MUTATED M5: $m5"; restore

if [ $METAL = 1 ]; then
  echo "== M6 the snapshot is Tensor::copy (a shallow clone on Metal)"
  perl -0pi -e 's/_ => self\.snapshot\[l\] = Some\(own_copy\(cur\)\?\),/_ => self.snapshot[l] = Some(cur.copy()?),/' $RS
  grep -q "Some(cur.copy()?)" $RS || echo "M6 did not apply"
  m6=$(run_metal); echo "MUTATED M6: $m6"; restore
fi

echo "== VERDICT"
echo "M1 wrong slot:        $m1 (RED = killed)"
[ $METAL = 1 ] && echo "M2 metal write offset: $m2 (RED = killed)"
[ $METAL = 1 ] && echo "M6 snapshot alias:     $m6 (RED = killed)"
echo "M3 keep KV lengths:   $m3 (RED = killed)"
echo "M4 stale cache row:   $m4 (RED = killed)"
echo "M5 accept everything: $m5 (RED = killed)"
