#!/bin/zsh
# Mutation harness for the (f16, f32, f16) routed matmul arm (issue #748,
# #750 item B; CONTRIBUTING.md §3.1a form). Two mutations, one per layer the
# arm owns:
# (G) the Metal `kernel_mul_mv_id_f16` plane decode — the destination plane
#     pointer is shifted by one element, so every f16 plane lands one row late;
#     the f16 routed tests must go RED (the f32 arm is untouched: its wrapper
#     is a different template).
# (H) the Rust host side — the f16 activation byte size is read as 4, so the
#     kernel's `src1` row strides are wrong; the f16 routed tests must go RED.
#
# §3.1a: clean arm FIRST; mutate IN PLACE; restore from a byte-copy (never
# `git checkout`); verify by SHA-256; key the verdict on the `test result:` line.
set -e
HERE=${0:A:h}
RS="$HERE/../src/kernels/kernels/quantized.rs"
METAL="$HERE/../src/kernels/metal_src/quantized.metal"
cd "$HERE/../.."

ARMS=(routed_f16_q4_0_gate_shape routed_f16_q4k_down_shape routed_f16_iq2_xxs_rows_not_multiple_of_tile routed_f16_q2_0_down_shape)
CONTROL=(routed_q4_0_gate_shape routed_q2_0_down_shape)

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
v=$(verdict "$(run_arms "${ARMS[@]}" "${CONTROL[@]}")")
echo "CLEAN: $v"
[[ "$v" == GREEN ]] || { echo "baseline broken; every result below is uninterpretable"; exit 1; }

echo "=== [2/5] mutation G: f16 plane pointer shifted by one element (Metal wrapper) ==="
perl -0pi -e 's/    device       half \* dst_cur  = dst \+ i1\*ne0 \+ i2\*ne1\*ne0;/    device       half * dst_cur  = dst + i1*ne0 + i2*ne1*ne0 + 1;/' "$METAL"
grep -q 'dst + i1\*ne0 + i2\*ne1\*ne0 + 1;' "$METAL" || { echo "MUTATE G: pattern not applied"; exit 1; }
echo "MUTATE G: applied"
echo "=== [3/5] mutation G: f16 arms RED, f32 control GREEN ==="
vg=$(verdict "$(run_arms "${ARMS[@]}")")
vgc=$(verdict "$(run_arms "${CONTROL[@]}")")
echo "MUTATED G: f16=$vg f32-control=$vgc"
restore
[[ "$(shasum -a 256 "$METAL" | cut -d' ' -f1)" == "$pre_m" ]] || { echo "restore after G failed"; exit 1; }

echo "=== [4/5] mutation H: f16 activation stride read as 4 bytes (host dispatch) ==="
perl -0pi -e 's/    let act_size = match triple\.activation\(\) \{\n        GemvFloat::F32 => core::mem::size_of::<f32>\(\),\n        GemvFloat::F16 => 2,\n    \} as u64;/    let act_size = match triple.activation() {\n        GemvFloat::F32 => core::mem::size_of::<f32>(),\n        GemvFloat::F16 => 4,\n    } as u64;/' "$RS"
grep -q 'GemvFloat::F16 => 4,' "$RS" || { echo "MUTATE H: pattern not applied"; exit 1; }
echo "MUTATE H: applied"
echo "=== [5/5] mutation H: f16 arms RED, f32 control GREEN ==="
vh=$(verdict "$(run_arms "${ARMS[@]}")")
vhc=$(verdict "$(run_arms "${CONTROL[@]}")")
echo "MUTATED H: f16=$vh f32-control=$vhc"

if [[ "$vg" == RED && "$vh" == RED && "$vgc" == GREEN && "$vhc" == GREEN ]]; then
  echo "VERDICT: both mutations killed on the f16 arm, the f32 arm untouched — the f16 routed tests are non-vacuous."
  exit 0
else
  echo "VERDICT: unexpected (G=$vg/$vgc H=$vh/$vhc) — a test that cannot fail is not a test."
  exit 1
fi
