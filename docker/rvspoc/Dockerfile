# SPDX-License-Identifier: Apache-2.0
# RVSPOC S2601 build environment: RISC-V cross-compile + QEMU user-mode runtime.
#
# Host: any (Linux x86_64, Linux arm64, or macOS Apple Silicon via Docker Desktop).
# Target: riscv64-linux-gnu with RVV 1.0 (RV64GCV).

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Base build deps + RISC-V cross toolchain (gcc-14 has full RVV 1.0 intrinsic support)
# qemu-user-static for running riscv64 binaries on the host arch.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
      build-essential \
      cmake \
      ninja-build \
      git \
      curl \
      wget \
      unzip \
      pkg-config \
      python3 \
      python3-pip \
      python3-venv \
      python3-dev \
      gcc-14-riscv64-linux-gnu \
      g++-14-riscv64-linux-gnu \
      qemu-user-static \
      protobuf-compiler \
      file \
      vim \
      less \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Symlink riscv64-linux-gnu-{gcc,g++} to the gcc-14 versions so plain
# CMake/Make invocations pick up the RVV-capable toolchain by default.
RUN ln -sf /usr/bin/riscv64-linux-gnu-gcc-14 /usr/local/bin/riscv64-linux-gnu-gcc \
 && ln -sf /usr/bin/riscv64-linux-gnu-g++-14 /usr/local/bin/riscv64-linux-gnu-g++

# Default workdir is the bind-mounted repo root.
WORKDIR /work

# Sanity: print toolchain versions on container start unless overridden.
CMD ["bash", "-lc", "riscv64-linux-gnu-gcc --version | head -1 && qemu-riscv64-static --version | head -1 && cmake --version | head -1 && ninja --version && echo 'rvspoc-s2601 build env ready.'"]
