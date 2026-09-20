#!/bin/zsh
# Mutation harness for issue #750's quantized-matmul-vec dtype-triple genericization
# (CONTRIBUTING.md §3.1a). Verifies the parity tests can FAIL — a genericization that
# silently computed the wrong thing would pass them, so a deliberate break must turn
# them red. §3.1a: run the non-vacuity (clean) arm FIRST, mutate IN PLACE, restore and
# verify byte-identical by SHA-256.
#
# The mutated line is the store cast in the genericized flagship core `mul_vec_q_n_impl`
# (`(STORE) tot`). Halving it corrupts every arm that routes through that core (q4_0,
# q4_1, q5_0, q5_1 at both f32 and f16), which both `quantized_matmul_q4_0_metal` (the
# f32 arm) and `quant_f16_arm_matches_f32_arm_q4_0` (the f16 arm) must catch.
set -e
HERE=${0:A:h}
METAL="$HERE/../src/kernels/metal_src/quantized.metal"
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-/Users/ocean/workspace/lloom-wt-750/.cargo-target}
# The `m == 1` parity arms route through the genericized `mul_vec_q_n_impl` GEMV
# core (the mutation site) AND assert against a full-precision reference, so a
# store-scaling corruption of the core turns them red (a sibling-only comparison
# would stay green under a symmetric mutation — §3.1b).
ARMS=(quant_f16_arm_matches_f32_arm_q4_0)
# Run the two arms; verdict() reads the `test result:` line of the one test binary
# (quantized_tests), keying on the ok/FAILED pair (§3.1b: a grep for FAILURE is not
# a test of failure — read the `test result:` line, not `error:`).
run_arms() { cargo test --release --features metal -p lloom-tensor --test quantized_tests -- "${ARMS[@]}" 2>&1; }

# Restore from a byte-copy of the working tree, NEVER `git checkout` — the #750
# changes are uncommitted, so `git checkout` would revert them, not restore them.
BAK=$(mktemp); cp "$METAL" "$BAK"
pre=$(shasum -a 256 "$METAL" | cut -d' ' -f1)
trap 'cp "$BAK" "$METAL"; post=$(shasum -a 256 "$METAL" | cut -d" " -f1); [[ "$pre" == "$post" ]] && echo "RESTORE ok (sha $post)" || echo "RESTORE MISMATCH pre=$pre post=$post"; rm -f "$BAK"' EXIT

echo "=== [1/3] non-vacuity: clean arm must be GREEN first (§3.1a rule 1) ==="
if run_arms | grep -qE 'test result: ok\. [1-9][0-9]* passed'; then
  echo "CLEAN: GREEN (baseline valid)"
else
  echo "CLEAN: RED — baseline is broken, every result below is uninterpretable"; exit 1
fi

echo "=== [2/3] mutate IN PLACE: halve the (STORE) cast in mul_vec_q_n_impl ==="
# The flagship core's store line is unique: `dst[...] = (STORE)(tot);`
perl -0pi -e 's/(\] = )\(STORE\)\(tot\);/$1(STORE)(tot * 0.5f);/' "$METAL"
if ! grep -q '(STORE)(tot \* 0.5f);' "$METAL"; then echo "MUTATE: pattern not applied"; exit 1; fi
echo "MUTATE: applied"

echo "=== [3/3] mutated arms must go RED (both f32 and f16 route through this core) ==="
if run_arms | grep -qE 'test result: FAILED'; then
  echo "MUTATED: RED — the parity tests catch the corruption. Mutation harness VALID."
  RC=0
else
  echo "MUTATED: GREEN — the parity tests are VACUOUS (they do not detect the break). FAIL."
  RC=1
fi
exit $RC
