#!/usr/bin/env bash
# Mutation harness for the QSA chunk prefill past the indexer's dense range
# (CONTRIBUTING §3.1a form): the clean arm FIRST; each mutant applied IN
# PLACE with perl and checked to have applied; restored from a byte copy
# taken at start, verified by SHA-256 and touched (cargo keys on mtime);
# the verdict from libtest's `test result:` lines and the killers named
# from its `failures:` list (a FAILED line never forms under --nocapture).
#
#   bash lloom/tests/mutate_746_qsa_chunk.sh
#
# M0  forward_embeds stops a chunk at the dense range again (the defect)
# M1  a chunk's later queries do not see the blocks its earlier tokens
#     completed (every row's visible blocks = the first sparse row's)
# M2  visible = (n_kv − 1) / ratio, an off-by-one
# M3  the tail one cell short (a query's own cell dropped)
# M4  the dense rows one too many (the first sparse query attends densely)
# M5  a pooled row's cells read as pool slots without its page table
set -u
cd "$(dirname "$0")/../.."   # candle/
QSA=lloom/src/models/qwen4exp/qsa.rs
MD=lloom/src/models/qwen4exp/model.rs
BAK=$(mktemp -d)
cp $QSA "$BAK/qsa.rs"; cp $MD "$BAK/model.rs"
sha() { shasum -a 256 "$1" | cut -d' ' -f1; }
S_QSA=$(sha $QSA); S_MD=$(sha $MD)
restore() {
  cp "$BAK/qsa.rs" $QSA; cp "$BAK/model.rs" $MD; touch $QSA $MD
  if [ "$(sha $QSA)" = "$S_QSA" ] && [ "$(sha $MD)" = "$S_MD" ]; then
    echo "RESTORE ok (sha ${S_QSA:0:12} ${S_MD:0:12})"
  else
    echo "RESTORE FAILED"; exit 2
  fi
}
trap restore EXIT

# One cargo run over every test of the mechanism, lib and integration, on
# Metal and the CPU arms alike — `--no-fail-fast`, or a failing lib binary
# stops cargo before the integration binary runs. Prints the verdict, then
# the killers.
run() {
  local out
  out=$(cargo test --release -p lloom --features metal --no-fail-fast --lib --test qwen4exp_chunk_past_dense \
        -- qsa::tests::chunk past_the_dense_range mtp_keys 2>&1)
  if echo "$out" | grep -q "could not compile"; then echo "BUILD-ERROR"; return; fi
  local results ran failed
  results=$(echo "$out" | grep -c "^test result:")
  ran=$(echo "$out" | grep "^test result:" | grep -oE "[0-9]+ passed; [0-9]+ failed" \
        | awk '{p+=$1; f+=$3} END {print p+f}')
  failed=$(echo "$out" | grep "^test result:" | grep -oE "[0-9]+ failed" | awk '{f+=$1} END {print f+0}')
  if [ "$results" -lt 2 ] || [ "${ran:-0}" -eq 0 ]; then echo "RAN-NOTHING (results $results)"; return; fi
  local killers
  killers=$(echo "$out" | awk '/^failures:$/ {f=1; next} f && /^    / {print $1} /^test result:/ {f=0}' \
            | sort -u | tr '\n' ' ')
  if [ "$failed" -eq 0 ]; then echo "GREEN ($ran tests)"; else echo "RED ($failed of $ran) killed by: $killers"; fi
}

apply() { # name, file, perl expression, the string that must then be present
  perl -0pi -e "$3" "$2"
  if ! grep -qF -- "$4" "$2"; then echo "$1 DID NOT APPLY"; restore; return 1; fi
}

echo "== CLEAN (must be GREEN, or nothing below is interpretable)"
clean=$(run); echo "CLEAN: $clean"
case "$clean" in GREEN*) ;; *) echo "clean arm not green; aborting"; exit 1 ;; esac

declare -a V
mut() { # name, file, perl, marker
  echo "== $1"
  if apply "$1" "$2" "$3" "$4"; then
    local r; r=$(run); echo "MUTATED $1: $r"; V+=("$1: $r"); restore
  else
    V+=("$1: NOT APPLIED")
  fi
}

mut "M0 the clamp at the dense range" $MD \
  's/let chunk_len = \|start: usize\| -> usize \{ \(seq - start\)\.min\(chunk\) \};/let chunk_len = |start: usize| -> usize { (seq - start).min(chunk).min(self.dense_limit().map_or(usize::MAX, |l| l.saturating_sub(start)).max(1)) };/' \
  'l.saturating_sub(start)).max(1)'
mut "M1 blocks completed mid-chunk invisible" $QSA \
  's/\.map\(\|t\| \(\(pos0 \+ t \+ 1\) \/ r\) as u32\)/.map(|_t| ((pos0 + 1) \/ r) as u32)/' \
  '.map(|_t| ((pos0 + 1) / r) as u32)'
mut "M2 visible off by one" $QSA \
  's/\.map\(\|t\| \(\(pos0 \+ t \+ 1\) \/ r\) as u32\)/.map(|t| ((pos0 + t) \/ r) as u32)/' \
  '.map(|t| ((pos0 + t) / r) as u32)'
mut "M3 tail one cell short" $QSA \
  's/n_keys\.push\(\(topk \* r \+ n_kv - start\) as u32\);/n_keys.push((topk * r + (n_kv - start).saturating_sub(1)) as u32);/' \
  '(n_kv - start).saturating_sub(1)'
mut "M4 one dense row too many" $QSA \
  's/Some\(ix\) => seq\.min\(ix\.dense_limit\(\)\.saturating_sub\(past\)\),/Some(ix) => seq.min((ix.dense_limit() + 1).saturating_sub(past)),/' \
  '(ix.dense_limit() + 1).saturating_sub(past)'
mut "M5 pooled cells without the page table" $QSA \
  's/pages: Some\(\(pool\.clone\(\), pt\)\),/pages: { let _ = pt; None },/' \
  'pages: { let _ = pt; None },'

echo "== VERDICT (RED = killed)"
for v in "${V[@]}"; do echo "$v"; done
