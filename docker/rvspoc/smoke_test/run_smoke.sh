#!/usr/bin/env bash
# Smoke test: cross-compile a tiny RVV intrinsic program and run it under qemu.
# Run inside the rvspoc-s2601 container (or any env with the right toolchain).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

echo "[1/3] Compiling rvv_hello.c with -march=rv64gcv ..."
riscv64-linux-gnu-gcc -O2 -march=rv64gcv -static rvv_hello.c -o rvv_hello.elf

echo "[2/3] File info:"
file rvv_hello.elf

echo "[3/3] Running under qemu (vlen=256) ..."
qemu-riscv64-static -cpu rv64,v=true,vlen=256,elen=64 ./rvv_hello.elf
