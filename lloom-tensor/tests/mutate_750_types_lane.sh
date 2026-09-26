#!/bin/zsh
# Mutation harness for #750 item A (the dtype axis on `Typed`) and item E's CPU
# half (the runtime-detected `sdot` arm). CONTRIBUTING §3.1a: the clean arms run
# FIRST, every mutation is applied IN PLACE and restored from a byte copy taken
# at harness start (never `git checkout`, which would revert uncommitted work),
# and each restore is verified by SHA-256.
#
# Verdicts read cargo's `test result:` lines (a grep for `error:` is not a test
# of failure, §3.1b): GREEN = every suite ran >= 1 test and none FAILED; RED =
# a suite reports FAILED; BUILD = a compile error with no test result; ZERO = a
# suite ran no test.
#
# Each cargo call takes its own `lloom-arb share` lease, so a queued measurement
# waits for one build rather than for the whole harness.
#
# Builds: A = default (`dotprod` compiled in); B = `-C target-feature=-dotprod`
# (the arm detected at run time); C = B with LLOOM_CPU_DISABLE=dotprod.
set -u
HERE=${0:A:h}
CANDLE=${HERE:h:h}
ROOT=${CANDLE:h}
SRC=$CANDLE/lloom-tensor/src
cd $CANDLE || exit 2
export CARGO_TERM_COLOR=never

typeset -A TD ENV
TD[A]=$ROOT/.cargo-target;           ENV[A]=""
TD[B]=$ROOT/.cargo-target/nodot;     ENV[B]="RUSTFLAGS=-Ctarget-feature=-dotprod"
TD[C]=$ROOT/.cargo-target/nodot;     ENV[C]="RUSTFLAGS=-Ctarget-feature=-dotprod LLOOM_CPU_DISABLE=dotprod"

typeset -A SUITE
SUITE[doc]="--doc typed"
SUITE[typed]="--test typed_tests"
SUITE[neon]="--lib -- cpu::arms quantized::neon"
SUITE[quant]="--test quantized_tests"

FILES=($SRC/typed.rs $SRC/quantized/neon.rs $SRC/cpu/arms.rs)
typeset -A BAK SHA
for f in $FILES; do
  BAK[$f]=$(mktemp); cp $f ${BAK[$f]}
  SHA[$f]=$(shasum -a 256 $f | cut -d' ' -f1)
done
restore_all() {
  for f in $FILES; do
    cp ${BAK[$f]} $f
    local now=$(shasum -a 256 $f | cut -d' ' -f1)
    [[ $now == ${SHA[$f]} ]] || { echo "RESTORE MISMATCH $f"; exit 3; }
  done
}
trap 'restore_all; echo "RESTORE ok (sha-256 verified, ${#FILES} files)"; for f in $FILES; do rm -f ${BAK[$f]}; done' EXIT

# run <build> <suite...>: one verdict over the named suites.
run() {
  local cfg=$1; shift
  local out verdict=GREEN s
  for s in $@; do
    out=$(lloom-arb share -- env CARGO_TARGET_DIR=${TD[$cfg]} ${=ENV[$cfg]} cargo test --release -p lloom-tensor ${=SUITE[$s]} 2>&1)
    local results=$(print -r -- "$out" | grep -E '^test result:')
    if print -r -- "$results" | grep -q FAILED; then
      verdict=RED
      print -r -- "$out" | grep -E '^test .* FAILED$|panicked at|compiled successfully|error\[E' | head -4 | sed 's/^/      /'
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

# mutate <file> <perl -0 substitution> <description>: apply, and prove it applied.
mutate() {
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

echo "=== clean arms first: every build must be GREEN before any RED below means anything"
run A doc typed neon quant; expect GREEN "clean, build A"
run B neon quant;           expect GREEN "clean, build B"
run C neon quant;           expect GREEN "clean, build C"
[[ $FAIL == 0 ]] || { echo "CLEAN ARM NOT GREEN: nothing below is interpretable"; exit 1; }

echo "=== M1  Typed::add accepts a view of any dtype"
mutate $SRC/typed.rs 's/pub fn add\(&self, rhs: &Typed<B, E>\)/pub fn add<E2: Element>(&self, rhs: &Typed<B, E2>)/' M1
run A doc; expect RED "the cross-dtype compile_fail doctest compiles"
restore_all

echo "=== M2  the f16 triple declares f32 activations"
mutate $SRC/typed.rs 's/type Act = half::f16;/type Act = f32;/' M2
run A doc typed; expect RED "the triple compile_fail doctest compiles, the legs test fails"
restore_all

# M3 (run 1, measurements/issue-750-raw/types-lane/mutation-run1.log) pinned
# the cross-dtype doctest to the wrong error code and it stayed GREEN: stable
# rustdoc does not check a `compile_fail` block's error code. The codes were
# dropped; the positive controls beside each compile-fail are what pin them.

echo "=== M4  TryFrom stops checking the dtype"
mutate $SRC/typed.rs 's/if !E::matches\(tensor\.dtype\(\)\)/if false \&\& !E::matches(tensor.dtype())/' M4
run A typed; expect RED "a wrong-dtype tensor is accepted at the boundary"
restore_all

echo "=== M5  vdotq_s32 computes an unsigned dot product (build A: compiled in)"
mutate $SRC/quantized/neon.rs 's/"sdot \{acc:v\}\.4s, \{a:v\}\.16b, \{b:v\}\.16b"/"udot {acc:v}.4s, {a:v}.16b, {b:v}.16b"/' M5
run A neon quant; expect RED "every single-row kernel is wrong"
echo "=== M5b the same mutation, build B: the runtime-detected arm is the one that runs"
run B neon quant; expect RED "the target_feature kernels execute sdot"
echo "=== M5c the same mutation, build C: the reference arm runs and never touches sdot"
run C neon quant; expect GREEN "LLOOM_CPU_DISABLE=dotprod routes around the broken arm"
restore_all

echo "=== M6  the second sdot of vdotq_s32_pair is unsigned"
mutate $SRC/quantized/neon.rs 's/"sdot \{acc:v\}\.4s, \{a1:v\}\.16b, \{b1:v\}\.16b"/"udot {acc:v}.4s, {a1:v}.16b, {b1:v}.16b"/' M6
run A neon; expect RED "the four-column kernels are wrong"
run A quant; echo "      (reported, not asserted: whether quantized_tests reaches vec_dot_4)"
restore_all

echo "=== M7  the q8_0 entry point takes the reference arm when sdot is native"
mutate $SRC/quantized/neon.rs 's/unsafe \{ vec_dot_q8_0_q8_0_sdot\(n, xs, ys\) \}\n    \} else \{\n        BlockQ8_0::vec_dot_unopt\(n, xs, ys\)/BlockQ8_0::vec_dot_unopt(n, xs, ys)\n    } else {\n        unsafe { vec_dot_q8_0_q8_0_sdot(n, xs, ys) }/' M7
run A neon; expect RED "the routing test sees the reference arm's bits"
run A quant; echo "      (reported: quantized_tests compares vec_dot to vec_dot_unopt, so it cannot see this)"
restore_all

echo "=== M8  the host set ignores LLOOM_CPU_DISABLE (build C)"
mutate $SRC/cpu/arms.rs 's/Features::detect\(\)\.0 & !disabled\.0/Features::detect().0 \& !(disabled.0 \& 0)/' M8
run C neon; expect RED "the disabled feature stays native"
restore_all

echo "=== done: $FAIL unexpected verdict(s)"
exit $FAIL
