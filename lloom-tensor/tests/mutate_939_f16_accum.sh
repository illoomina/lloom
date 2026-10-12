#!/bin/zsh
# Mutation harness for the CPU f16 matmul's accumulator axis (lloom #939,
# `cpu/f16_matmul.rs`; CONTRIBUTING.md §3.1a). The parity tests compare each
# arm with a scalar model of the rounding it declares, bit for bit; a kernel
# that silently computed another rounding would have to turn them red. Three
# breaks, each applied in place and restored byte-identical (SHA-256, then
# `touch` so cargo rebuilds the clean source):
#
#   M1  the f16 chains never promote (`PROMOTE_EVERY = usize::MAX`): one f16
#       chain of k per lane — the shape the axis exists to retire. Kills
#       `each_arm_computes_its_declared_rounding_bit_for_bit` (the f16/16
#       model has the promotion) and the floor test at k >= 512.
#   M2  the f32 arm's lane tree reordered ((l0+l2)+(l1+l3)) + ...: the same
#       sum in another order. Kills the bit-for-bit test on both arms.
#   M3  the scalar tail dropped (k % 8 products ignored). Kills the bit-for-bit
#       test at k = 23, 15, 136 and the f32 arm's once-rounded bound.
#
# Run from anywhere; a target dir of its own so the mutant never lands in a
# shared build:  CARGO_TARGET_DIR=/tmp/lloom939-mutate zsh mutate_939_f16_accum.sh
set -u
HERE=${0:A:h}
SRC="$HERE/../src/cpu/f16_matmul.rs"
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-/tmp/lloom939-mutate}
cd "$HERE/../.."

run_arms() {
  cargo nextest run -p lloom-tensor -E 'test(cpu_f16_matmul_accum::)' 2>&1
}
# nextest's Summary line is the verdict (§3.1b: key on the runner's own
# summary, not on `error:`); a run of zero tests is its own verdict.
verdict() {
  local out="$1"
  local s
  s=$(echo "$out" | grep -E 'Summary \[' | tail -1)
  if [[ -z "$s" ]]; then echo "NORUN"; return; fi
  if echo "$s" | grep -qE ' 0 tests run'; then echo "NORUN"; return; fi
  if echo "$s" | grep -qE '[1-9][0-9]* failed'; then echo "RED"; else echo "GREEN"; fi
}

BAK=$(mktemp)
cp "$SRC" "$BAK"
pre=$(shasum -a 256 "$SRC" | cut -d' ' -f1)
restore() {
  cp "$BAK" "$SRC"
  touch "$SRC"
  local post
  post=$(shasum -a 256 "$SRC" | cut -d' ' -f1)
  if [[ "$pre" == "$post" ]]; then echo "RESTORE ok (sha $post)"; else echo "RESTORE MISMATCH pre=$pre post=$post"; exit 2; fi
}
trap 'restore; rm -f "$BAK"' EXIT

echo "=== [0] clean arm first (§3.1a rule 1) ==="
out=$(run_arms); v=$(verdict "$out")
echo "$out" | grep -E 'PASS|FAIL|Summary'
echo "CLEAN: $v"
if [[ "$v" != "GREEN" ]]; then echo "the clean arm is not green: nothing below is interpretable"; exit 1; fi

st=0
mutate() { # name perl-expr grep-check
  local name=$1 expr=$2 check=$3
  echo "=== [$name] ==="
  cp "$BAK" "$SRC"
  perl -0pi -e "$expr" "$SRC"
  if ! grep -qF -- "$check" "$SRC"; then echo "$name: MUTATE pattern not applied"; st=1; return; fi
  touch "$SRC"
  out=$(run_arms); v=$(verdict "$out")
  echo "$out" | grep -E 'PASS|FAIL|Summary'
  echo "$name: $v"
  if [[ "$v" != "RED" ]]; then echo "$name: the tests did not catch it"; st=1; fi
}

mutate M1 's/const PROMOTE_EVERY: usize = 16;/const PROMOTE_EVERY: usize = usize::MAX;/' 'const PROMOTE_EVERY: usize = usize::MAX;
        const NAME: &'"'"'static str = "f16/16";'
mutate M2 's/let s = \(\(l\[0\] \+ l\[1\]\) \+ \(l\[2\] \+ l\[3\]\)\) \+ \(\(l\[4\] \+ l\[5\]\) \+ \(l\[6\] \+ l\[7\]\)\);/let s = ((l[0] + l[2]) + (l[1] + l[3])) + ((l[4] + l[6]) + (l[5] + l[7]));/' '((l[0] + l[2]) + (l[1] + l[3]))'
mutate M3 's/for kk in k8\.\.k \{\n        t = /for kk in k..k {\n        t = /' 'for kk in k..k {'

restore
trap - EXIT
rm -f "$BAK"
if [[ $st == 0 ]]; then echo "MUTATION HARNESS: every break killed"; else echo "MUTATION HARNESS: a break survived"; fi
exit $st
