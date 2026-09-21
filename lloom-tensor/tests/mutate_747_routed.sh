#!/bin/zsh
# Mutation harness for the routed-expert quantized matmul dispatch (issue #747;
# CONTRIBUTING.md §3.1a). Two mutations, one per layer the dispatch owns:
# (C) the Rust host side — the expert byte stride `nb02` in
#     `call_quantized_matmul_mv_id_t` is off by one block, so every expert
#     but the first is read from the wrong offset; every routed arm must go RED.
# (D) the Metal Q2_0 core's dot (the same mutation as `mutate_747_q2_0.sh` A),
#     which must reach the routed Q2_0 arms THROUGH `kernel_mul_mv_id_q2_0_f32`
#     — proving the `_id` instantiation wraps the core the plain arm tests.
#
# §3.1a: clean arm FIRST; mutate IN PLACE; restore from a byte-copy (never
# `git checkout`); verify by SHA-256; key the verdict on the `test result:` line.
set -e
HERE=${0:A:h}
RS="$HERE/../src/kernels/kernels/quantized.rs"
METAL="$HERE/../src/kernels/metal_src/quantized.metal"
cd "$HERE/../.."

ARMS_C=(routed_q4_0_gate_shape routed_iq4_nl_down_shape routed_q2_0_top10_over_64_experts)
ARMS_D=(routed_q2_0_gate_shape routed_q2_0_down_shape)

run_arms() { cargo test --release --features metal -p lloom-tensor --test moe_routed_metal -- "$@" 2>&1; }
verdict() {
  local out="$1"
  if echo "$out" | grep -qE 'test result: ok\. [1-9][0-9]* passed'; then echo GREEN
  elif echo "$out" | grep -qE 'test result: FAILED'; then echo RED
  else echo NORUN; fi
}

BAK_RS=$(mktemp); cp "$RS" "$BAK_RS"
BAK_M=$(mktemp); cp "$METAL" "$BAK_M"
pre_rs=$(shasum -a 256 "$RS" | cut -d' ' -f1)
pre_m=$(shasum -a 256 "$METAL" | cut -d' ' -f1)
restore() { cp "$BAK_RS" "$RS"; cp "$BAK_M" "$METAL"; }
trap 'restore; post_rs=$(shasum -a 256 "$RS" | cut -d" " -f1); post_m=$(shasum -a 256 "$METAL" | cut -d" " -f1); if [[ "$pre_rs" == "$post_rs" && "$pre_m" == "$post_m" ]]; then echo "RESTORE ok (rs $post_rs, metal $post_m)"; else echo "RESTORE MISMATCH"; fi; rm -f "$BAK_RS" "$BAK_M"' EXIT

echo "=== [1/5] non-vacuity: clean arms must be GREEN first (§3.1a rule 1) ==="
v=$(verdict "$(run_arms "${ARMS_C[@]}" "${ARMS_D[@]}")")
echo "CLEAN: $v"
[[ "$v" == GREEN ]] || { echo "baseline broken; every result below is uninterpretable"; exit 1; }

echo "=== [2/5] mutation C: expert stride nb02 off by one block (host dispatch) ==="
perl -0pi -e 's/let nb02 = \(n \* \(k \/ dtype_block_size\(dtype\)\) \* dtype_type_size\(dtype\)\) as u64;/let nb02 = ((n * (k \/ dtype_block_size(dtype)) + 1) * dtype_type_size(dtype)) as u64;/' "$RS"
grep -q '+ 1) \* dtype_type_size(dtype)) as u64;' "$RS" || { echo "MUTATE C: pattern not applied"; exit 1; }
echo "MUTATE C: applied"
echo "=== [3/5] mutation C arms must go RED ==="
vc=$(verdict "$(run_arms "${ARMS_C[@]}")")
echo "MUTATED C: $vc"
restore
[[ "$(shasum -a 256 "$RS" | cut -d' ' -f1)" == "$pre_rs" ]] || { echo "restore after C failed"; exit 1; }

echo "=== [4/5] mutation D: halve the high-bit weight in the Q2_0 dot (Metal core) ==="
perl -0pi -e 's/return qb_curr->d \* \(acc_lo \+ 2\.0f \* acc_hi - sumy\);/return qb_curr->d * (acc_lo + 1.0f * acc_hi - sumy);/' "$METAL"
grep -q 'acc_lo + 1.0f \* acc_hi - sumy' "$METAL" || { echo "MUTATE D: pattern not applied"; exit 1; }
echo "MUTATE D: applied"
echo "=== [5/5] mutation D arms (routed Q2_0) must go RED ==="
vd=$(verdict "$(run_arms "${ARMS_D[@]}")")
echo "MUTATED D: $vd"

if [[ "$vc" == RED && "$vd" == RED ]]; then
  echo "VERDICT: both mutations killed — the routed matmul tests are non-vacuous."
  exit 0
else
  echo "VERDICT: a mutation SURVIVED (C=$vc D=$vd) — a test that cannot fail is not a test."
  exit 1
fi
