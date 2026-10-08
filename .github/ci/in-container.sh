#!/usr/bin/env bash
# Run a command in the project's container image (the Dockerfile beside
# this tree's Cargo.toml), the way the workflows do, on any host with Docker:
#
#   .github/ci/in-container.sh [--stage test|dev|cuda] [--build] -- <cmd...>
#
#   .github/ci/in-container.sh -- cpu-tests.sh lloom 290 token_ring_fuzz
#   .github/ci/in-container.sh --stage cuda -- cuda-positive-control.sh
#   .github/ci/in-container.sh --build -- cargo nextest run -p lloom-engine
#
# The paths are the engine tree's: `candle/` in the development repository,
# the repository root in the public one. That tree is mounted at /work and
# the command runs there, with its gate scripts (`.github/ci/`) on PATH
# inside the image; nothing outside it is visible, as nothing outside it is
# published. The cargo registry and the target directory are mounted from
# the host, so a second run is incremental and CI can cache them:
#
#   LLOOM_CI_REGISTRY   host path for /usr/local/cargo/registry
#                       (default: a Docker volume, `lloom-ci-registry`)
#   LLOOM_CI_TARGET     host path for /work/target
#                       (default: a Docker volume, `lloom-ci-target-<stage>`)
#   LLOOM_CI_IMAGE      image tag to use (default: `lloom-ci:<stage>`)
#   DOCKER              the docker binary (default: `docker`)
#
# `--build` rebuilds the image from the Dockerfile; without it the image is
# built only when the tag is missing. A host whose docker needs a different
# binary or a prefix sets DOCKER (the NAS: `DOCKER=/Volume3/@apps/DockerEngine/dockerd/bin/docker`;
# the Pi: `DOCKER="sudo docker"`).
set -euo pipefail

stage=test
build=0
while [ $# -gt 0 ]; do
  case "$1" in
    --stage) stage="$2"; shift 2 ;;
    --build) build=1; shift ;;
    --) shift; break ;;
    *) echo "usage: $0 [--stage test|dev|cuda] [--build] -- <cmd...>" >&2; exit 2 ;;
  esac
done
if [ $# -eq 0 ]; then
  echo "usage: $0 [--stage test|dev|cuda] [--build] -- <cmd...>" >&2
  exit 2
fi

# The engine tree: the directory holding this script's `.github/`.
root="$(cd "$(dirname "$0")/../.." && pwd)"
# DOCKER may carry a prefix (`DOCKER="sudo docker"` where the socket is
# root's, as on the Pi), so it is split on spaces.
read -ra docker <<< "${DOCKER:-docker}"
image="${LLOOM_CI_IMAGE:-lloom-ci:${stage}}"

if [ "$build" -eq 1 ] || ! "${docker[@]}" image inspect "$image" >/dev/null 2>&1; then
  echo "=== building ${image} (stage ${stage}) from $root/Dockerfile ==="
  "${docker[@]}" build --target "$stage" -t "$image" -f "$root/Dockerfile" "$root"
fi

registry="${LLOOM_CI_REGISTRY:-lloom-ci-registry}"
target="${LLOOM_CI_TARGET:-lloom-ci-target-${stage}}"
case "$registry" in /*) mkdir -p "$registry" ;; esac
case "$target" in /*) mkdir -p "$target" ;; esac

# Pass the variables a gate reads through (the proptest case count, the
# fixture selector, the probe switches), never the host's whole environment.
env_args=()
for v in PROPTEST_CASES LLOOM_FIXTURE LLOOM_PROBE_DUMP RAYON_NUM_THREADS CARGO_TERM_COLOR; do
  if [ -n "${!v:-}" ]; then env_args+=(-e "$v=${!v}"); fi
done

# The container runs as the calling user, so what it writes into the
# mounted target and registry is the host user's: a root-owned target broke
# the runner's cache save (`tar` exit 2) and left every CI run cold. The
# official rust image keeps CARGO_HOME world-writable for exactly this.
exec "${docker[@]}" run --rm \
  --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$root:/work" \
  -v "$registry:/usr/local/cargo/registry" \
  -v "$target:/work/target" \
  -w /work \
  "${env_args[@]}" \
  "$image" "$@"
