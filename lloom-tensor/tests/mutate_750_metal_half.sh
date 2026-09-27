#!/bin/zsh
# Mutation harness for #750 item E's Metal half (the native-arm registry and
# the triple dispatch gate) and the read-only quant refusal. CONTRIBUTING
# §3.1a: the clean arms run FIRST, every mutation is applied IN PLACE and
# restored from a byte copy taken at harness start (never `git checkout`), each
# restore is verified by SHA-256 and then touched, so cargo cannot reuse the
# mutant's build for the restored source (a restored inode keeps its old mtime).
#
# Verdicts read cargo's `test result:` lines: GREEN = every suite ran >= 1 test
# and none FAILED; RED = a suite reports FAILED; BUILD = no test result; ZERO =
# a suite ran no test. Each cargo call takes its own `lloom-arb share` lease,
# unless MUTATE_IN_LEASE=1 says the whole harness already runs inside one.
#
# Configurations: M = `--features metal`; D = M with LLOOM_METAL_DISABLE=f16,
# the device set with `f16` withdrawn; P = no `metal` feature (the CPU build).
set -u
HERE=${0:A:h}
CANDLE=${HERE:h:h}
ROOT=${CANDLE:h}
SRC=$CANDLE/lloom-tensor/src
cd $CANDLE || exit 2
export CARGO_TERM_COLOR=never

typeset -A FEAT ENV
FEAT[M]="--features metal"; ENV[M]=""
FEAT[D]="--features metal"; ENV[D]="LLOOM_METAL_DISABLE=f16"
FEAT[P]="";                 ENV[P]=""

typeset -A SUITE
SUITE[arms]="--lib -- arms::tests"
SUITE[gate]="--lib -- kernels::kernels::quantized::arm_tests"
SUITE[quant]="--test quantized_tests"
SUITE[f16arm]="--test quantized_tests -- quant_f16_arm_matches_f32_arm"
SUITE[ro]="--test quantized_tests -- read_only"

FILES=($SRC/arms.rs $SRC/cpu/arms.rs $SRC/kernels/metal/arms.rs
       $SRC/kernels/kernels/quantized.rs $SRC/quantized/mod.rs)
typeset -A BAK SHA
for f in $FILES; do
  BAK[$f]=$(mktemp); cp $f ${BAK[$f]}
  SHA[$f]=$(shasum -a 256 $f | cut -d' ' -f1)
done
restore_all() {
  for f in $FILES; do
    cp ${BAK[$f]} $f
    touch $f
    local now=$(shasum -a 256 $f | cut -d' ' -f1)
    [[ $now == ${SHA[$f]} ]] || { echo "RESTORE MISMATCH $f"; exit 3; }
  done
}
trap 'restore_all; echo "RESTORE ok (sha-256 verified, ${#FILES} files)"; for f in $FILES; do rm -f ${BAK[$f]}; done' EXIT

run() {  # run <config> <suite...>: one verdict over the named suites
  local cfg=$1; shift
  local out verdict=GREEN s
  for s in $@; do
    local arb=(lloom-arb share --); [[ -n ${MUTATE_IN_LEASE-} ]] && arb=()
    out=$($arb env ${=ENV[$cfg]} cargo test --release -p lloom-tensor ${=FEAT[$cfg]} ${=SUITE[$s]} 2>&1)
    local results=$(print -r -- "$out" | grep -E '^test result:')
    if print -r -- "$results" | grep -q FAILED; then
      verdict=RED
      print -r -- "$out" | grep -E '^test .* FAILED$|panicked at|error\[E' | head -4 | sed 's/^/      /'
    elif [[ -z $results ]]; then
      [[ $verdict == RED ]] || verdict=BUILD
      print -r -- "$out" | grep -E '^error' | head -3 | sed 's/^/      /'
    elif ! print -r -- "$results" | grep -qE 'ok\. [1-9][0-9]* passed'; then
      [[ $verdict == RED ]] || verdict=ZERO
    fi
    echo "    [$cfg:$s] $(print -r -- "$results" | sed -E 's/; 0 ignored.*//' | tr '\n' ' ')"
  done
  VERDICT=$verdict
}

mutate() {  # mutate <file> <perl -0 substitution> <label>: apply, and prove it applied
  local f=$1 expr=$2
  perl -0pi -e "$expr" $f
  if [[ $(shasum -a 256 $f | cut -d' ' -f1) == ${SHA[$f]} ]]; then
    echo "  NOT APPLIED: $3"; exit 4
  fi
}

typeset -i FAIL=0
expect() {  # expect <want> <label>
  if [[ $VERDICT == $1 ]]; then echo "  => $VERDICT (expected $1)  $2"
  else echo "  => $VERDICT (EXPECTED $1)  $2"; FAIL+=1; fi
}

echo "=== clean arms first: every configuration must be GREEN before any RED below means anything"
run M arms gate quant; expect GREEN "clean, metal"
run D arms gate;       expect GREEN "clean, f16 withdrawn: the refusal tests take their refusing branch"
run P arms ro;         expect GREEN "clean, no metal"
[[ $FAIL == 0 ]] || { echo "CLEAN ARM NOT GREEN: nothing below is interpretable"; exit 1; }

echo "=== the gate is on the production path: withdraw f16 and the QMatMul f16 arm is refused"
run D f16arm; expect RED "quant_f16_arm_matches_f32_arm_* reach the refusal through QMatMul::forward"

echo "=== M1  apple7 registers no f16"
mutate $SRC/kernels/metal/arms.rs 's/\(Feature::Apple7, &\[Feature::F16\]\)/(Feature::Apple7, \&[])/' M1
run M arms f16arm; expect RED "the registry test and every f16 arm through QMatMul"
restore_all

echo "=== M2  the triple check skips the device"
mutate $SRC/kernels/kernels/quantized.rs 's/        arms::require_all\(device, self\.requires\(\), \|\| format!\("\{op\} \{\}", self\.label\(\)\)\)\?;\n//' M2
run M gate; expect RED "the int8 arm is refused as unimplemented, not by the feature it lacks"
run D gate; expect RED "the withdrawn f16 arm is dispatched"
restore_all

echo "=== M3  require_all passes an arm whose features the device lacks"
mutate $SRC/kernels/metal/arms.rs 's/    if have\.contains_all\(needed\) \{/    if true || have.contains_all(needed) {/' M3
run M arms gate; expect RED "int8-mma is not refused"
restore_all

echo "=== M4  detection asks no family"
mutate $SRC/kernels/metal/arms.rs 's/\.filter\(\|f\| f\.family\(\)\.is_some_and\(\|family\| raw\.supportsFamily\(family\)\)\)/.filter(|f| f.family().is_some_and(|_| false))/' M4
run M arms; expect RED "the family set disagrees with the architecture"
restore_all

echo "=== M5  the device set ignores LLOOM_METAL_DISABLE"
mutate $SRC/kernels/metal/arms.rs 's/\.minus\(Features::from_env\("LLOOM_METAL_DISABLE"\)\)/.minus(Features::empty())/' M5
run D arms gate; expect RED "the withdrawn f16 stays native"
restore_all

echo "=== M6  a refusal names a capability the device has"
mutate $SRC/arms.rs 's/let missing = needed\.minus\(self\);/let missing = needed;/' M6
run P arms; expect RED "the refusal text lists present capabilities"
restore_all

echo "=== M7  IQ4_NL is not read-only"
mutate $SRC/quantized/mod.rs 's/                \| Self::IQ4NL\n//' M7
run P ro; expect RED "quantizing to IQ4_NL reaches the from_float panic"
restore_all

echo "=== M8  the CPU matmul of a read-only type is not refused"
mutate $SRC/quantized/mod.rs 's/(fn matmul_t\(&self[^\n]*\n)        T::DTYPE\.refuse_read_only\("the CPU matmul of"\)\?;\n/$1/' M8
run P ro; expect RED "the IQ vec_dot panic is reached"
restore_all

echo "=== done: $FAIL unexpected verdict(s)"
exit $FAIL
