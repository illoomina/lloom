#!/usr/bin/env bash
# The shape probes stay out of the serving binary (zero cost to release):
# `shape-probe` is enabled only from `lloom-fuzz` (its own workspace) and from
# `lloom`'s self dev-dependency, so no feature of `lloom-engine` may pull it
# in. `lloom`'s own tree carries it through that dev-dependency, which is the
# positive control: a grep that cannot find it there is blind, not green.
#
# Resolves the graph only; builds nothing.
set -euo pipefail

# Run from the workspace root (two levels above this script) regardless of
# caller CWD.
cd "$(dirname "$0")/../.."

engine=$(cargo tree -e features -p lloom-engine --all-features | grep -c 'feature "shape-probe"' || true)
control=$(cargo tree -e features -p lloom | grep -c 'feature "shape-probe"' || true)
echo "shape-probe in lloom-engine's feature tree (--all-features): ${engine}"
echo "shape-probe in lloom's own tree (positive control):          ${control}"
if [ "${control}" -eq 0 ]; then
  echo "FAIL: the positive control found nothing -- the check cannot see the feature" >&2
  exit 1
fi
if [ "${engine}" -ne 0 ]; then
  echo "FAIL: shape-probe reaches lloom-engine; the path:" >&2
  cargo tree -e features -p lloom-engine --all-features -i lloom-tensor | grep -B4 'shape-probe' >&2 || true
  exit 1
fi
echo "ok"
