#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Build the LTOIR linking tools
#
# Prerequisites:
#   - CUDA Toolkit (libNVVM, nvJitLink)
#   - gcc
#
# Usage:
#   ./build_tools.sh

set -e

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

echo "=== Building Small Type FFI Tools ==="
echo "CUDA_HOME: $CUDA_HOME"
echo ""

# Build compile_ltoir (libNVVM)
echo "Building compile_ltoir..."
gcc -o compile_ltoir compile_ltoir.c \
    -I${CUDA_HOME}/nvvm/include \
    -L${CUDA_HOME}/nvvm/lib64 -lnvvm \
    -Wl,-rpath,${CUDA_HOME}/nvvm/lib64
echo "  ✓ compile_ltoir"

# Build link_ltoir (nvJitLink)
echo "Building link_ltoir..."
gcc -o link_ltoir link_ltoir.c \
    -I${CUDA_HOME}/include \
    -L${CUDA_HOME}/lib64 -lnvJitLink \
    -Wl,-rpath,${CUDA_HOME}/lib64
echo "  ✓ link_ltoir"

echo ""
echo "=== Build Complete ==="
echo ""
echo "Tools:"
echo "  ./compile_ltoir  - Compile LLVM IR to LTOIR (libNVVM -gen-lto)"
echo "  ./link_ltoir     - Link multiple LTOIR files (nvJitLink)"
