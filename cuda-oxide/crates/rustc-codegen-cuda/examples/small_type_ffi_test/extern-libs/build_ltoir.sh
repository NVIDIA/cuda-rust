#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
# Build external device functions to LTOIR for cuda-oxide FFI testing
#
# This script compiles CUDA C++ source files to LTOIR (Link-Time Optimization IR)
# which can be linked with cuda-oxide kernels via nvJitLink.
#
# Usage:
#   ./build_ltoir.sh [arch]
#
# Arguments:
#   arch - Target GPU architecture (default: sm_120)
#
# Output:
#   *.ltoir      - Binary LTOIR files (for nvJitLink)
#   *_text.ltoir - Text LTOIR files (for inspection/debugging)
#   *.o          - Object files (contain LTOIR)
# =============================================================================

set -e  # Exit on any error

# Parse architecture argument (default: sm_120 for Blackwell)
ARCH="${1:-sm_120}"
NVCC_CCBIN="${NVCC_CCBIN:-${CUDAHOSTCXX:-}}"
NVCC_FLAGS=()
if [[ -n "$NVCC_CCBIN" ]]; then
    NVCC_FLAGS+=("-ccbin=$NVCC_CCBIN")
fi

echo "Building for architecture: $ARCH"
if [[ -n "$NVCC_CCBIN" ]]; then
    echo "nvcc host compiler: $NVCC_CCBIN"
fi
echo ""

# Setup nvvm-tools path for nvvm-dis (converts binary LTOIR to text)
# These tools are optional - only needed for text LTOIR generation
NVVM_TOOLS="${NVVM_TOOLS_NEXT:-$HOME/dev/nvvm-tools-next}/Linux_amd64_release"
NVVM_DIS="$NVVM_TOOLS/nvvm-dis"
export LD_LIBRARY_PATH="$NVVM_TOOLS:$LD_LIBRARY_PATH"

# =============================================================================
# compile_ltoir: Compile a single CUDA file to LTOIR
#
# Arguments:
#   $1 - Source file (e.g., "small_type_funcs.cu")
#
# The key nvcc flags are:
#   -dc    : Compile to relocatable device code (enables separate compilation)
#   -dlto  : Enable device link-time optimization (generates LTOIR)
#   --keep : Keep intermediate files (including .ltoir)
# =============================================================================
compile_ltoir() {
    local src="$1"
    local base="${src%.cu}"

    echo "=== Compiling $src ==="
    # -dc: relocatable device code, -dlto: device LTO, --keep: retain .ltoir
    nvcc "${NVCC_FLAGS[@]}" -arch=$ARCH -dc -dlto --keep "$src" -o "${base}.o" 2>&1

    if [ -f "${base}.ltoir" ]; then
        echo "  Binary LTOIR: ${base}.ltoir ($(wc -c < ${base}.ltoir) bytes)"

        # Optionally convert binary LTOIR to text format for debugging.
        # This is best-effort: an nvvm-dis older than the CUDA toolkit cannot
        # parse newer LTOIR, and that must not fail the build (set -e).
        if [ -x "$NVVM_DIS" ]; then
            if "$NVVM_DIS" "${base}.ltoir" > "${base}_text.ltoir" 2>&1; then
                echo "  Text LTOIR:   ${base}_text.ltoir ($(wc -c < ${base}_text.ltoir) bytes)"
            else
                echo "  Text LTOIR:   skipped (nvvm-dis cannot parse this LTOIR)"
                rm -f "${base}_text.ltoir"
            fi
        fi
    else
        echo "  ERROR: LTOIR not generated for $src"
        return 1
    fi
    echo ""
}

# Compile small scalar ABI device functions
compile_ltoir "small_type_funcs.cu"

# Clean up intermediate files
echo "=== Cleaning up intermediate files ==="
rm -f *.ii *.cudafe* *.fatbin *.fatbin.c *.ptx *.module_id
echo "Done."
echo ""

# Summary
echo "=== Generated Files ==="
echo ""
echo "Binary LTOIR (for nvJitLink):"
ls -la *.ltoir 2>/dev/null | grep -v "_text.ltoir" || true
echo ""
echo "Text LTOIR (for inspection):"
ls -la *_text.ltoir 2>/dev/null || true
echo ""
echo "Object files:"
ls -la *.o 2>/dev/null || true
echo ""

# Show exported functions
echo "=== Exported Functions ==="
for ltoir in *_text.ltoir; do
    if [ -f "$ltoir" ]; then
        echo ""
        echo "From ${ltoir%_text.ltoir}.cu:"
        grep "^  define.*@[a-z]" "$ltoir" | sed 's/.*@\([a-zA-Z_][a-zA-Z0-9_]*\).*/  \1/' | sort -u
    fi
done
