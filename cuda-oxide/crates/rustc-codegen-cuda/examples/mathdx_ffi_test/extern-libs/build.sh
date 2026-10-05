#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
# Build MathDx C++ wrappers to LTOIR for linking with cuda-oxide kernels
#
# This script compiles CUDA C++ source files to LTOIR (Link-Time Optimization IR)
# which can be linked with cuda-oxide kernels via nvJitLink.
#
# Usage:
#   ./build.sh [options] [arch]
#
# Options:
#   --test    Also build and link test_separate.cubin for CUDA C++ validation
#   --clean   Remove all generated files before building
#   --help    Show this help message
#
# Arguments:
#   arch - Target GPU architecture (default: sm_120)
#
# Prerequisites:
#   - CUDA Toolkit 12.x+ with nvcc
#   - MathDx (cuBLASDx, cuFFTDx) installed
#
# Environment variables:
#   MATHDX_ROOT: Path to MathDx installation (default: auto-detected)
#
# Output:
#   *.ltoir      - Binary LTOIR files (for nvJitLink)
#   *_text.ltoir - Text LTOIR files (for inspection/debugging)
#
# Examples:
#   ./build.sh                    # Build LTOIR for sm_120
#   ./build.sh sm_90              # Build LTOIR for sm_90
#   ./build.sh --test             # Build LTOIR + test cubin
#   ./build.sh --clean --test     # Clean and rebuild everything
# =============================================================================

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# =============================================================================
# Parse arguments
# =============================================================================
BUILD_TEST=false
DO_CLEAN=false
ARCH="sm_120"

while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            BUILD_TEST=true
            shift
            ;;
        --clean)
            DO_CLEAN=true
            shift
            ;;
        --help|-h)
            head -35 "$0" | tail -33
            exit 0
            ;;
        sm_*)
            ARCH="$1"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "=== MathDx LTOIR Build Script ==="
echo "Architecture: $ARCH"
echo "Build test:   $BUILD_TEST"
echo ""

NVCC_CCBIN="${NVCC_CCBIN:-${CUDAHOSTCXX:-}}"
NVCC_FLAGS=()
if [[ -n "$NVCC_CCBIN" ]]; then
    NVCC_FLAGS+=("-ccbin=$NVCC_CCBIN")
    echo "nvcc host compiler: $NVCC_CCBIN"
    echo ""
fi

# =============================================================================
# Clean if requested
# =============================================================================
if [ "$DO_CLEAN" = true ]; then
    echo "=== Cleaning generated files ==="
    rm -f *.ltoir *.o *.ii *.cudafe* *.fatbin *.fatbin.c *.ptx *.module_id *.gpu
    rm -f test_separate.cubin test_separate
    echo "Done."
    echo ""
fi

# =============================================================================
# Setup paths
# =============================================================================

# Detect MathDx installation
if [ -z "$MATHDX_ROOT" ]; then
    # Check common installation paths
    if [ -d "/usr/local/mathdx" ]; then
        MATHDX_ROOT="/usr/local/mathdx"
    elif [ -d "/opt/nvidia/mathdx" ]; then
        MATHDX_ROOT="/opt/nvidia/mathdx"
    else
        echo "Error: MathDx not found. Set MATHDX_ROOT environment variable."
        echo "Download MathDx from: https://developer.nvidia.com/cublasdx-downloads"
        exit 1
    fi
fi

MATHDX_INCLUDE="${MATHDX_ROOT}/include"
CUTLASS_INCLUDE="${MATHDX_ROOT}/external/cutlass/include"

# Verify include paths exist
if [ ! -d "$MATHDX_INCLUDE" ]; then
    echo "Error: MathDx include directory not found: $MATHDX_INCLUDE"
    exit 1
fi

if [ ! -d "$CUTLASS_INCLUDE" ]; then
    echo "Error: CUTLASS include directory not found: $CUTLASS_INCLUDE"
    exit 1
fi

echo "MathDx: $MATHDX_ROOT"
echo ""

# Setup nvvm-tools path for nvvm-dis (converts binary LTOIR to text)
NVVM_TOOLS="${NVVM_TOOLS_NEXT:-$HOME/dev/nvvm-tools-next}/Linux_amd64_release"
NVVM_DIS="$NVVM_TOOLS/nvvm-dis"
export LD_LIBRARY_PATH="$NVVM_TOOLS:$LD_LIBRARY_PATH"

# =============================================================================
# compile_ltoir: Compile a single CUDA file to LTOIR
# =============================================================================
compile_ltoir() {
    local src="$1"
    local base="${src%.cu}"
    local extra_flags="${2:-}"

    echo "=== Compiling $src ==="
    
    # -dc: relocatable device code, -dlto: device LTO, --keep: retain .ltoir
    # --expt-relaxed-constexpr: REQUIRED for cuBLASDx - enables constexpr in device code
    # -Wno-deprecated-declarations: suppress MathDx deprecation warnings
    nvcc "${NVCC_FLAGS[@]}" -arch=$ARCH -dc -dlto --keep -std=c++17 --expt-relaxed-constexpr \
        -Wno-deprecated-declarations \
        -I"${MATHDX_INCLUDE}" -I"${CUTLASS_INCLUDE}" \
        $extra_flags "$src" -o "${base}.o" 2>&1

    if [ -f "${base}.ltoir" ]; then
        echo "  Binary LTOIR: ${base}.ltoir ($(wc -c < ${base}.ltoir) bytes)"

        # Optionally convert binary LTOIR to text format for debugging. This is
        # a convenience only -- the binary LTOIR above is what nvJitLink
        # consumes. An nvvm-dis older than the toolkit that produced the LTOIR
        # fails with an NvvmIRVersion/LlvmVersion mismatch; that must not abort
        # the build (wrapping in `if` also keeps `set -e` from tripping).
        if [ -x "$NVVM_DIS" ]; then
            if "$NVVM_DIS" "${base}.ltoir" > "${base}_text.ltoir" 2>&1; then
                echo "  Text LTOIR:   ${base}_text.ltoir ($(wc -c < ${base}_text.ltoir) bytes)"
            else
                echo "  Text LTOIR:   skipped (nvvm-dis incompatible with this LTOIR version)"
                rm -f "${base}_text.ltoir"
            fi
        fi
    else
        echo "  ERROR: LTOIR not generated for $src"
        return 1
    fi
    echo ""
}

# =============================================================================
# Build MathDx wrappers
# =============================================================================

# Build cuFFTDx wrappers
if [ -f "cufftdx_wrappers.cu" ]; then
    compile_ltoir "cufftdx_wrappers.cu"
fi

# Build cuFFTDx function wrappers (separate file for debug functions)
if [ -f "cufftdx_wrappers_funcs.cu" ]; then
    compile_ltoir "cufftdx_wrappers_funcs.cu"
fi

# Build cuBLASDx wrappers
# NOTE: cuBLASDx does NOT work on Blackwell (sm_120) as of MathDx 25.12
# See bugs/BUG-004-cublasdx-blackwell-sm120-unsupported.md
if [ -f "cublasdx_wrappers.cu" ]; then
    # cuBLASDx is unsupported on Blackwell (sm_120) as of MathDx 25.12, and the
    # cuFFTDx FFI test does not depend on it; tolerate a compile failure here so
    # the build still succeeds on architectures where cuBLASDx won't compile.
    compile_ltoir "cublasdx_wrappers.cu" \
        || echo "  (skipped cublasdx_wrappers: compile failed; unsupported on $ARCH)"
fi

# Build test kernels (for verifying CUDA C++ extern behavior)
if [ -f "cuda_test_kernels.cu" ]; then
    compile_ltoir "cuda_test_kernels.cu"
fi

# =============================================================================
# Clean up intermediate files
# =============================================================================
echo "=== Cleaning up intermediate files ==="
rm -f *.ii *.cudafe* *.fatbin *.fatbin.c *.ptx *.module_id *.gpu *.o
echo "Done."
echo ""

# =============================================================================
# Build test cubin (optional)
# =============================================================================
if [ "$BUILD_TEST" = true ]; then
    echo "=== Building test_separate.cubin ==="
    
    TOOLS_DIR="$SCRIPT_DIR/../tools"
    if [ -x "$TOOLS_DIR/link_ltoir" ]; then
        # Collect LTOIR files to link
        LTOIR_FILES="cuda_test_kernels.ltoir cufftdx_wrappers_funcs.ltoir"
        
        # Add cuBLASDx wrappers if they exist
        if [ -f "cublasdx_wrappers.ltoir" ]; then
            LTOIR_FILES="$LTOIR_FILES cublasdx_wrappers.ltoir"
            echo "  Including cuBLASDx wrappers"
        fi
        
        "$TOOLS_DIR/link_ltoir" -arch=${ARCH} -o test_separate.cubin $LTOIR_FILES
        echo "  Created: test_separate.cubin"
        
        # Build host test program
        if [ -f "test_separate.cu" ]; then
            echo ""
            echo "=== Building test_separate host program ==="
            nvcc "${NVCC_FLAGS[@]}" -arch=$ARCH test_separate.cu -o test_separate -lcuda
            echo "  Created: test_separate"
            echo ""
            echo "Run './test_separate' to validate CUDA C++ extern calls"
        fi
    else
        echo "  WARNING: link_ltoir tool not found at $TOOLS_DIR/link_ltoir"
        echo "  Build it with: cd ../tools && ./build_tools.sh"
    fi
    echo ""
fi

# =============================================================================
# Summary
# =============================================================================
echo "=== Generated Files ==="
echo ""
echo "Binary LTOIR (for nvJitLink):"
ls -la *.ltoir 2>/dev/null | grep -v "_text.ltoir" || echo "  (none)"
echo ""
echo "Text LTOIR (for inspection):"
ls -la *_text.ltoir 2>/dev/null || echo "  (none)"
echo ""

if [ "$BUILD_TEST" = true ] && [ -f "test_separate.cubin" ]; then
    echo "Test files:"
    ls -la test_separate.cubin test_separate 2>/dev/null || true
    echo ""
fi

# Show exported functions
echo "=== Exported Functions ==="
for ltoir in *_text.ltoir; do
    if [ -f "$ltoir" ]; then
        echo ""
        echo "From ${ltoir%_text.ltoir}.cu:"
        grep "^  define.*@[a-z]" "$ltoir" 2>/dev/null | sed 's/.*@\([a-zA-Z_][a-zA-Z0-9_]*\).*/  \1/' | sort -u || echo "  (no functions found)"
    fi
done
