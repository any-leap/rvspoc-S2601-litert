#!/usr/bin/env bash
# Build the RVSPOC S2601 cross-compile Docker image.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
docker build -t rvspoc-s2601:latest "$HERE"
