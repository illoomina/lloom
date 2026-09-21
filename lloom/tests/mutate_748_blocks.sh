#!/bin/zsh
# Mutation harness for the #748 blocks (CONTRIBUTING.md §3.1a form): one
# mutation per novel piece, each must turn its parity test RED.
#   (A) DeltaNet Kernel B — the pairwise decay matrix loses its exp (decay
#       becomes the raw log-difference): `chunk_forward_matches_sequential_decode`
#       must go RED (the block-level `prefill_matches_decode_replay` with it).
#   (A2) DeltaNet Kernel B — the unit-lower solve replaced by the nilpotent
#       doubling product (exact in arithmetic, catastrophic in fp32):
#       `chunk_forward_is_stable_on_near_identical_keys` must go RED.
#   (B) N-gram PLE hash — the EOS cut is dropped (older predecessors reach the
#       hash): `eos_resets_the_window_by_scope` must go RED.
#   (C) PLE hash — the remainder is left C-style (negative on a negative mix):
#       `remainder_is_python_style_on_negative_mix` must go RED.
#   (D) QSA indexer — the score's ReLU is dropped (a plain dot):
#       `selection_matches_scalar_reference` must go RED.
#   (E) QSA rope — adjacent-pair rotation instead of NEOX:
#       `partial_rope_is_neox_paired` must go RED.
#   (F) QSA chunk attention — the causal mask is dropped:
#       `prefill_matches_decode_replay` (qsa) must go RED.
#
# §3.1a: clean arm FIRST; mutate IN PLACE; restore from a byte-copy (never
# `git checkout`); verify by SHA-256; key the verdict on the `test result:` line.
set -e
HERE=${0:A:h}
DN="$HERE/../src/models/qwen4exp/deltanet.rs"
PLE="$HERE/../src/models/qwen4exp/ple.rs"
QSA="$HERE/../src/models/qwen4exp/qsa.rs"
cd "$HERE/../.."

run_tests() { cargo test -p lloom --lib -- "$@" 2>&1; }
verdict() {
  local out="$1"
  if echo "$out" | grep -qE 'test result: ok\. [1-9][0-9]* passed'; then echo GREEN
  elif echo "$out" | grep -qE 'test result: FAILED'; then echo RED
  else echo NORUN; fi
}

typeset -A BAK PRE
for f in "$DN" "$PLE" "$QSA"; do
  BAK[$f]=$(mktemp); cp "$f" "${BAK[$f]}"
  PRE[$f]=$(shasum -a 256 "$f" | cut -d' ' -f1)
done
restore() { for f in "$DN" "$PLE" "$QSA"; do cp "${BAK[$f]}" "$f"; done; }
trap 'restore; ok=1; for f in "$DN" "$PLE" "$QSA"; do post=$(shasum -a 256 "$f" | cut -d" " -f1); [[ "$post" == "${PRE[$f]}" ]] || ok=0; done; if [[ $ok == 1 ]]; then echo "RESTORE ok (sha ${PRE[$DN]:0:8} ${PRE[$PLE]:0:8} ${PRE[$QSA]:0:8})"; else echo "RESTORE MISMATCH"; fi; for f in "$DN" "$PLE" "$QSA"; do rm -f "${BAK[$f]}"; done' EXIT

ALL=(chunk_forward_matches_sequential_decode chunk_forward_is_stable_on_near_identical_keys prefill_matches_decode_replay eos_resets_the_window_by_scope remainder_is_python_style_on_negative_mix selection_matches_scalar_reference partial_rope_is_neox_paired)

echo "=== [1/7] non-vacuity: clean arms must be GREEN first (§3.1a rule 1) ==="
v=$(verdict "$(run_tests "${ALL[@]}")")
echo "CLEAN: $v"
[[ "$v" == GREEN ]] || { echo "baseline broken; every result below is uninterpretable"; exit 1; }

mutate() {
  # mutate <file> <perl-expr> <grep-check> <label>
  perl -0pi -e "$2" "$1"
  grep -qF -- "$3" "$1" || { echo "MUTATE $4: pattern not applied"; exit 1; }
  echo "MUTATE $4: applied"
}
check_restore() { [[ "$(shasum -a 256 "$1" | cut -d' ' -f1)" == "${PRE[$1]}" ]] || { echo "restore of $1 failed"; exit 1; }; }

echo "=== [2/7] mutation A: Kernel B decay without the exp ==="
mutate "$DN" 's/let decay = lower\.where_cond\(&diff, &neg\)\?\.exp\(\)\?;/let decay = lower.where_cond(\&diff, \&neg)?;/' 'let decay = lower.where_cond(&diff, &neg)?;' A
va=$(verdict "$(run_tests chunk_forward_matches_sequential_decode)")
echo "MUTATED A: $va"
restore; check_restore "$DN"

echo "=== [2b/7] mutation A2: the unit-lower solve as the doubling product ==="
mutate "$DN" 's/    let t = unit_lower_inverse\(&a\)\?;/    let t = { let eye = Tensor::eye(chunk, dt, dev)?.broadcast_as((h_v, nc, chunk, chunk))?.contiguous()?; let mut bp = a.neg()?; let mut t = (\&eye + \&bp)?; let mut span = 2usize; while span < chunk { bp = bp.matmul(\&bp)?; t = t.matmul(\&(\&eye + \&bp)?)?; span *= 2; } t };/' 'let mut bp = a.neg()?;' A2
va2=$(verdict "$(run_tests chunk_forward_is_stable_on_near_identical_keys)")
echo "MUTATED A2: $va2"
restore; check_restore "$DN"

echo "=== [3/7] mutation B: PLE hash without the EOS cut ==="
mutate "$PLE" 's/            crossed \|= candidate == self\.eos;/            crossed |= false \&\& candidate == self.eos;/' 'crossed |= false && candidate == self.eos;' B
vb=$(verdict "$(run_tests eos_resets_the_window_by_scope)")
echo "MUTATED B: $vb"
restore; check_restore "$PLE"

echo "=== [4/7] mutation C: PLE remainder left C-style ==="
mutate "$PLE" 's/                if rem < 0 \{\n                    rem \+= size;\n                \}/                if rem < 0 { rem += 0; }/' 'if rem < 0 { rem += 0; }' C
vc=$(verdict "$(run_tests remainder_is_python_style_on_negative_mix)")
echo "MUTATED C: $vc"
restore; check_restore "$PLE"

echo "=== [5/7] mutation D: indexer score without the ReLU ==="
mutate "$QSA" 's/k32\.matmul\(&q32\)\?\.relu\(\)\?\.sum\(1\)\? \/\/ \[visible\]/k32.matmul(\&q32)?.sum(1)? \/\/ [visible]/' 'k32.matmul(&q32)?.sum(1)? // [visible]' D
vd=$(verdict "$(run_tests selection_matches_scalar_reference)")
echo "MUTATED D: $vd"
restore; check_restore "$QSA"

echo "=== [6/7] mutation E: QSA rope pairs adjacent dims ==="
mutate "$QSA" 's/    let x_lo = x\.narrow\(2, 0, half\)\?;\n    let x_hi = x\.narrow\(2, half, half\)\?;/    let (seq, h, _) = x.dims3()?;\n    let pairs = x.narrow(2, 0, n_rot)?.reshape((seq, h, half, 2))?;\n    let x_lo = pairs.narrow(3, 0, 1)?.reshape((seq, h, half))?;\n    let x_hi = pairs.narrow(3, 1, 1)?.reshape((seq, h, half))?;/' 'let pairs = x.narrow(2, 0, n_rot)?.reshape((seq, h, half, 2))?;' E
ve=$(verdict "$(run_tests partial_rope_is_neox_paired)")
echo "MUTATED E: $ve"
restore; check_restore "$QSA"

echo "=== [7/7] mutation F: QSA chunk attention without the causal mask ==="
mutate "$QSA" 's/            m\[t \* kv_len \+ j\] = 1;/            m[t * kv_len + j] = 1; let _ = j;\n        }\n        for j in 0..kv_len {\n            m[t * kv_len + j] = 1;/' 'm[t * kv_len + j] = 1; let _ = j;' F
vf=$(verdict "$(run_tests qsa::tests::prefill_matches_decode_replay)")
echo "MUTATED F: $vf"
restore; check_restore "$QSA"

if [[ "$va" == RED && "$va2" == RED && "$vb" == RED && "$vc" == RED && "$vd" == RED && "$ve" == RED && "$vf" == RED ]]; then
  echo "VERDICT: all seven mutations killed — the #748 block tests are non-vacuous."
  exit 0
else
  echo "VERDICT: a mutation SURVIVED (A=$va A2=$va2 B=$vb C=$vc D=$vd E=$ve F=$vf) — a test that cannot fail is not a test."
  exit 1
fi
