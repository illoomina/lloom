#!/bin/zsh
# Mutation harness for the Q2_0 Metal kernels (issue #747; CONTRIBUTING.md §3.1a).
# Two independent mutations, because the GEMV core and the dequant table are
# separate code: (A) the fork's dot product `d * (acc_lo + 2*acc_hi - sumy)` has
# its high-bit weight halved, which corrupts every arm through
# `kernel_mul_mv_q2_0_impl` (f32 GEMV, f16 GEMV, and the routed `mul_mv_id`);
# (B) `dequantize_q2_0`'s `(q - 1) * d` becomes `(q - 0.5) * d`, which corrupts
# the prefill GEMM and the row gather. Each mutation's arm set must go RED.
#
# §3.1a: clean arm FIRST; mutate IN PLACE; restore from a byte-copy (never
# `git checkout`, which reverts uncommitted work) and verify by SHA-256; key the
# verdict on the `test result:` line, not on `error:`.
set -e
HERE=${0:A:h}
METAL="$HERE/../src/kernels/metal_src/quantized.metal"
cd "$HERE/../.."

ARMS_A=(q2_0_gemv_f32_matches_cpu_dequant_reference q2_0_gemv_f16_arm_matches_cpu_dequant_reference)
ARMS_B=(q2_0_gemm_prefill_matches_cpu_dequant_reference q2_0_embedding_matches_cpu_dequant_reference)

run_arms() { cargo test --release --features metal -p lloom-tensor --test q2_0_metal -- "$@" 2>&1; }
verdict() {
  # prints GREEN / RED / NORUN from the one `test result:` line
  local out="$1"
  if echo "$out" | grep -qE 'test result: ok\. [1-9][0-9]* passed'; then echo GREEN
  elif echo "$out" | grep -qE 'test result: FAILED'; then echo RED
  else echo NORUN; fi
}

BAK=$(mktemp); cp "$METAL" "$BAK"
pre=$(shasum -a 256 "$METAL" | cut -d' ' -f1)
restore() { cp "$BAK" "$METAL"; }
trap 'restore; post=$(shasum -a 256 "$METAL" | cut -d" " -f1); [[ "$pre" == "$post" ]] && echo "RESTORE ok (sha $post)" || echo "RESTORE MISMATCH pre=$pre post=$post"; rm -f "$BAK"' EXIT

echo "=== [1/5] non-vacuity: clean arms must be GREEN first (§3.1a rule 1) ==="
v=$(verdict "$(run_arms "${ARMS_A[@]}" "${ARMS_B[@]}")")
echo "CLEAN: $v"
[[ "$v" == GREEN ]] || { echo "baseline broken; every result below is uninterpretable"; exit 1; }

echo "=== [2/5] mutation A: halve the high-bit weight in the Q2_0 dot ==="
perl -0pi -e 's/return qb_curr->d \* \(acc_lo \+ 2\.0f \* acc_hi - sumy\);/return qb_curr->d * (acc_lo + 1.0f * acc_hi - sumy);/' "$METAL"
grep -q 'acc_lo + 1.0f \* acc_hi - sumy' "$METAL" || { echo "MUTATE A: pattern not applied"; exit 1; }
echo "MUTATE A: applied"
echo "=== [3/5] mutation A arms (f32 GEMV, f16 GEMV) must go RED ==="
va=$(verdict "$(run_arms "${ARMS_A[@]}")")
echo "MUTATED A: $va"
restore
[[ "$(shasum -a 256 "$METAL" | cut -d' ' -f1)" == "$pre" ]] || { echo "restore after A failed"; exit 1; }

echo "=== [4/5] mutation B: dequantize_q2_0 (q - 1) -> (q - 0.5) ==="
perl -0pi -e 's/reg_f\[i\]\[0\] = \(\(float\)\(\(b >> 0\) & 3\) - 1\.0f\) \* d;/reg_f[i][0] = ((float)((b >> 0) \& 3) - 0.5f) * d;/' "$METAL"
grep -q '& 3) - 0.5f) \* d;' "$METAL" || { echo "MUTATE B: pattern not applied"; exit 1; }
echo "MUTATE B: applied"
echo "=== [5/5] mutation B arms (prefill GEMM, row gather) must go RED ==="
vb=$(verdict "$(run_arms "${ARMS_B[@]}")")
echo "MUTATED B: $vb"

if [[ "$va" == RED && "$vb" == RED ]]; then
  echo "VERDICT: both mutations killed — the Q2_0 parity tests are non-vacuous."
  exit 0
else
  echo "VERDICT: a mutation SURVIVED (A=$va B=$vb) — a test that cannot fail is not a test."
  exit 1
fi
