#!/usr/bin/env bash
# Drop into an interactive shell in the RVSPOC build env, with the repo
# bind-mounted at /work. Pass a command to run non-interactively.
#
# Usage:
#   ./docker/rvspoc/run.sh                    # interactive shell
#   ./docker/rvspoc/run.sh bash -c "make foo" # one-shot command
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"

# Use -it only if stdin is a TTY (so CI/scripted callers still work).
TTY_FLAGS=()
if [ -t 0 ] && [ -t 1 ]; then
  TTY_FLAGS=(-it)
else
  TTY_FLAGS=(-i)
fi

if [ $# -eq 0 ]; then
  exec docker run --rm "${TTY_FLAGS[@]}" \
    -v "$REPO_ROOT:/work" \
    -w /work \
    rvspoc-s2601:latest \
    bash
else
  exec docker run --rm "${TTY_FLAGS[@]}" \
    -v "$REPO_ROOT:/work" \
    -w /work \
    rvspoc-s2601:latest \
    "$@"
fi
