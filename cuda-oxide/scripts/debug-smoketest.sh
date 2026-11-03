#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# scripts/debug-smoketest.sh -- end-to-end cuda-gdb validation of device
# debug info (CUDA_OXIDE_DEBUG=full).
#
# Builds full-debug fixtures and checks their DWARF in cuda-gdb on a real GPU.
#
# Main checks:
#   caller PC -> test_option
#   helper PC -> get_mut -> test_option
#   pointer stack homes -> live addresses/pointees + null control
#   AS1 / AS3 / AS4 -> Rust lookup + type + value
#   brkpt -> debugger stop -> continue -> completed kernel
#
# compiler_features also covers closures, enums, and supported projections.
# Generic examples check their source breakpoint, stack, arguments, and locals.
#
# This requires cuda-gdb and a working NVIDIA GPU. Missing prerequisites are a
# SKIP (exit 0), so callers must check them before treating this as a CI gate.
#
# Usage:
#   scripts/debug-smoketest.sh            # default example (compiler_features)
#   scripts/debug-smoketest.sh debug_pointer_locals
#   scripts/debug-smoketest.sh device_global
#   scripts/debug-smoketest.sh constant_memory_simple
#   scripts/debug-smoketest.sh shared_debug
#   scripts/debug-smoketest.sh debug
#   BREAK_AT=vecadd scripts/debug-smoketest.sh vecadd  # generic example
#   CUDA_OXIDE_TARGET=sm_90 scripts/debug-smoketest.sh   # pin the arch
#   CUDA_OXIDE_DEBUG_LOG_DIR=/tmp/cuda-gdb-logs \
#       scripts/debug-smoketest.sh compiler_features     # retain complete logs

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXAMPLE="${1:-compiler_features}"
EXAMPLE_DIR="$REPO_ROOT/crates/rustc-codegen-cuda/examples/$EXAMPLE"

CUDA_GDB="${CUDA_OXIDE_CUDA_GDB:-$(command -v cuda-gdb || echo /usr/local/cuda/bin/cuda-gdb)}"

skip() {
    echo "debug-smoketest: SKIP ($1)"
    exit 0
}

[ -x "$CUDA_GDB" ] || skip "cuda-gdb not found (set CUDA_OXIDE_CUDA_GDB)"
command -v nvidia-smi >/dev/null 2>&1 || skip "nvidia-smi not found"
nvidia-smi -L >/dev/null 2>&1 || skip "no usable NVIDIA GPU / driver"
[ -d "$EXAMPLE_DIR" ] || { echo "debug-smoketest: FAIL (no example '$EXAMPLE')"; exit 1; }

# Resolve the device arch: explicit override wins, else the local GPU's cc.
if [ -n "${CUDA_OXIDE_TARGET:-}" ]; then
    ARCH="$CUDA_OXIDE_TARGET"
else
    CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ')"
    [ -n "$CC" ] || skip "could not read compute capability"
    ARCH="sm_${CC}"
fi

echo "debug-smoketest: example=$EXAMPLE arch=$ARCH"

# Build with full device debug info.
( cd "$REPO_ROOT" && CUDA_OXIDE_DEBUG=full CUDA_OXIDE_TARGET="$ARCH" \
    cargo oxide build "$EXAMPLE" ) || { echo "debug-smoketest: FAIL (build)"; exit 1; }

BIN="$EXAMPLE_DIR/target/release/$EXAMPLE"
[ -x "$BIN" ] || { echo "debug-smoketest: FAIL (no binary at $BIN)"; exit 1; }

if [ "$EXAMPLE" = "debug_pointer_locals" ]; then
    bash "$EXAMPLE_DIR/verify-debug-info.sh" \
        "$EXAMPLE_DIR/debug_pointer_locals.ll" \
        || { echo "debug-smoketest: FAIL (raw-pointer LLVM debug-info shape)"; exit 1; }
    ( cd "$EXAMPLE_DIR" && "./target/release/$EXAMPLE" ) \
        || { echo "debug-smoketest: FAIL (raw-pointer fixture execution)"; exit 1; }
elif [ "$EXAMPLE" = "device_global" ]; then
    bash "$EXAMPLE_DIR/verify-debug-info.sh" \
        || { echo "debug-smoketest: FAIL (device-global LLVM debug-info shape)"; exit 1; }
elif [ "$EXAMPLE" = "compiler_features" ]; then
    bash "$EXAMPLE_DIR/verify-debug-info.sh" \
        || { echo "debug-smoketest: FAIL (caller/helper cubin DWARF ranges)"; exit 1; }
elif [ "$EXAMPLE" = "constant_memory_simple" ]; then
    CUDA_OXIDE_VERIFY_CUDA_GDB=1 bash "$EXAMPLE_DIR/verify-debug-info.sh" \
        || { echo "debug-smoketest: FAIL (constant-memory AS4 debug contract)"; exit 1; }
    echo "debug-smoketest: PASS (constant-memory AS4 graph, cubin address class, Rust lookup, and value verified on $ARCH)"
    exit 0
elif [ "$EXAMPLE" = "shared_debug" ]; then
    bash "$EXAMPLE_DIR/verify-debug-info.sh" \
        || { echo "debug-smoketest: FAIL (shared-memory AS3 debug contract)"; exit 1; }
elif [ "$EXAMPLE" = "debug" ]; then
    bash "$EXAMPLE_DIR/verify-code-shape.sh" \
        || { echo "debug-smoketest: FAIL (debug-intrinsic line-table contract)"; exit 1; }
fi

export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

# Drive cuda-gdb: stop at a kernel, walk to the kernel frame, dump args/locals.
# The default remains self-cleaning for ordinary use. Validation runs set an
# explicit directory so every complete transcript survives for manual review;
# truncate the known names up front so a repeated run cannot pass by reading a
# stale transcript from an earlier session.
if [ -n "${CUDA_OXIDE_DEBUG_LOG_DIR:-}" ]; then
    mkdir -p "$CUDA_OXIDE_DEBUG_LOG_DIR" \
        || { echo "debug-smoketest: FAIL (cannot create log directory $CUDA_OXIDE_DEBUG_LOG_DIR)"; exit 1; }
    DEBUG_LOG_DIR="$(cd "$CUDA_OXIDE_DEBUG_LOG_DIR" && pwd)"
    LOG_STEM="${EXAMPLE//[^[:alnum:]_.-]/_}"
    GDB_LOG="$DEBUG_LOG_DIR/${LOG_STEM}-main.cuda-gdb.log"
    CLOSURE_GDB_LOG="$DEBUG_LOG_DIR/${LOG_STEM}-closure.cuda-gdb.log"
    ENUM_GDB_LOG="$DEBUG_LOG_DIR/${LOG_STEM}-enum.cuda-gdb.log"
    PROJECTION_GDB_LOG="$DEBUG_LOG_DIR/${LOG_STEM}-projection.cuda-gdb.log"
    RUNTIME_INDEX_GDB_LOG="$DEBUG_LOG_DIR/${LOG_STEM}-runtime-index.cuda-gdb.log"
    ENUM_PROJECTION_GDB_LOG="$DEBUG_LOG_DIR/${LOG_STEM}-enum-projection.cuda-gdb.log"
    DEREF_GDB_LOG="$DEBUG_LOG_DIR/${LOG_STEM}-deref.cuda-gdb.log"
    INLINE_CALLER_GDB_LOG="$DEBUG_LOG_DIR/${LOG_STEM}-inline-caller.cuda-gdb.log"
    INLINE_HELPER_GDB_LOG="$DEBUG_LOG_DIR/${LOG_STEM}-inline-helper.cuda-gdb.log"
    for log in "$GDB_LOG" "$CLOSURE_GDB_LOG" "$ENUM_GDB_LOG" "$PROJECTION_GDB_LOG" \
               "$RUNTIME_INDEX_GDB_LOG" "$ENUM_PROJECTION_GDB_LOG" "$DEREF_GDB_LOG" \
               "$INLINE_CALLER_GDB_LOG" "$INLINE_HELPER_GDB_LOG"; do
        : >"$log" || { echo "debug-smoketest: FAIL (cannot write log $log)"; exit 1; }
    done
    echo "debug-smoketest: retaining complete cuda-gdb logs in $DEBUG_LOG_DIR"
else
    GDB_LOG="$(mktemp)"
    CLOSURE_GDB_LOG="$(mktemp)"
    ENUM_GDB_LOG="$(mktemp)"
    PROJECTION_GDB_LOG="$(mktemp)"
    RUNTIME_INDEX_GDB_LOG="$(mktemp)"
    ENUM_PROJECTION_GDB_LOG="$(mktemp)"
    DEREF_GDB_LOG="$(mktemp)"
    INLINE_CALLER_GDB_LOG="$(mktemp)"
    INLINE_HELPER_GDB_LOG="$(mktemp)"
    trap 'rm -f "$GDB_LOG" "$CLOSURE_GDB_LOG" "$ENUM_GDB_LOG" "$PROJECTION_GDB_LOG" "$RUNTIME_INDEX_GDB_LOG" "$ENUM_PROJECTION_GDB_LOG" "$DEREF_GDB_LOG" "$INLINE_CALLER_GDB_LOG" "$INLINE_HELPER_GDB_LOG"' EXIT
fi

fail=0

# Run from the example dir: the host binary resolves its embedded device
# artifact relative to the working directory.
if [ "$EXAMPLE" = "debug_pointer_locals" ]; then
    DEBUG_SOURCE="$EXAMPLE_DIR/src/main.rs"
    POINTER_MARKER="CUDA_OXIDE_DEBUG_POINTER_BREAKPOINT"
    POINTER_LINE="$(grep -nF "$POINTER_MARKER" "$DEBUG_SOURCE" | head -1 | cut -d: -f1)"
    [ -n "$POINTER_LINE" ] \
        || { echo "debug-smoketest: FAIL (raw-pointer breakpoint marker not found)"; exit 1; }

    ( cd "$EXAMPLE_DIR" && timeout 300 "$CUDA_GDB" --batch \
        -ex 'set pagination off' \
        -ex 'set language auto' \
        -ex 'set breakpoint pending on' \
        -ex "break $DEBUG_SOURCE:$POINTER_LINE" \
        -ex 'run' \
        -ex 'show language' \
        -ex 'frame 0' \
        -ex 'info args' \
        -ex 'info locals' \
        -ex 'echo CUDA_OXIDE_PTR_TYPE\n' \
        -ex 'whatis ptr' \
        -ex 'echo CUDA_OXIDE_FPTR_TYPE\n' \
        -ex 'whatis fptr' \
        -ex 'echo CUDA_OXIDE_NULL_PTR_TYPE\n' \
        -ex 'whatis null_ptr' \
        -ex 'printf "CUDA_OXIDE_PTR_ADDRESS=%p\n", ptr' \
        -ex 'printf "CUDA_OXIDE_FPTR_ADDRESS=%p\n", fptr' \
        -ex 'printf "CUDA_OXIDE_PTR_POINTEE=%d\n", *ptr' \
        -ex 'printf "CUDA_OXIDE_FPTR_POINTEE=%g\n", *fptr' \
        -ex 'printf "CUDA_OXIDE_NULL_PTR=%p\n", null_ptr' \
        -ex 'backtrace' \
        -ex 'kill' \
        "./target/release/$EXAMPLE" ) >"$GDB_LOG" 2>&1
elif [ "$EXAMPLE" = "device_global" ]; then
    DEBUG_SOURCE="$EXAMPLE_DIR/src/main.rs"
    GLOBAL_MARKER="CUDA_OXIDE_DEBUG_GLOBAL_BREAKPOINT"
    GLOBAL_LINE="$(grep -nF "$GLOBAL_MARKER" "$DEBUG_SOURCE" | head -1 | cut -d: -f1)"
    [ -n "$GLOBAL_LINE" ] \
        || { echo "debug-smoketest: FAIL (device-global breakpoint marker not found)"; exit 1; }

    ( cd "$EXAMPLE_DIR" && timeout 300 "$CUDA_GDB" --batch \
        -ex 'set pagination off' \
        -ex 'set language auto' \
        -ex 'set breakpoint pending on' \
        -ex "break $DEBUG_SOURCE:$GLOBAL_LINE" \
        -ex 'run' \
        -ex 'show language' \
        -ex 'frame 0' \
        -ex 'printf "CUDA_OXIDE_DEVICE_COUNTER=%llu\n", device_global::DEVICE_COUNTER' \
        -ex 'printf "CUDA_OXIDE_DEVICE_MARKER=%u\n", device_global::DEVICE_MARKER' \
        -ex 'printf "CUDA_OXIDE_GLOBAL_LEFT=%u\n", device_global::debug_left::SAME_LEAF' \
        -ex 'printf "CUDA_OXIDE_GLOBAL_RIGHT=%llu\n", device_global::debug_right::SAME_LEAF' \
        -ex 'ptype device_global::DEVICE_COUNTER' \
        -ex 'ptype device_global::DEVICE_MARKER' \
        -ex 'ptype device_global::debug_left::SAME_LEAF' \
        -ex 'ptype device_global::debug_right::SAME_LEAF' \
        -ex 'backtrace' \
        -ex 'kill' \
        "./target/release/$EXAMPLE" ) >"$GDB_LOG" 2>&1
elif [ "$EXAMPLE" = "shared_debug" ]; then
    DEBUG_SOURCE="$EXAMPLE_DIR/src/main.rs"
    SHARED_MARKER="DEBUG_SHARED_BREAK"
    SHARED_LINE="$(grep -nF "$SHARED_MARKER" "$DEBUG_SOURCE" | head -1 | cut -d: -f1)"
    [ -n "$SHARED_LINE" ] \
        || { echo "debug-smoketest: FAIL (shared-memory breakpoint marker not found)"; exit 1; }

    ( cd "$EXAMPLE_DIR" && timeout 300 "$CUDA_GDB" --batch \
        -ex 'set pagination off' \
        -ex 'set language auto' \
        -ex 'set breakpoint pending on' \
        -ex "break $DEBUG_SOURCE:$SHARED_LINE" \
        -ex 'run' \
        -ex 'show language' \
        -ex 'frame 0' \
        -ex 'info locals' \
        -ex 'whatis TILE' \
        -ex 'ptype TILE' \
        -ex 'printf "CUDA_OXIDE_SHARED_SIZE=%u\n", sizeof(TILE)' \
        -ex 'printf "CUDA_OXIDE_SHARED_BLOCK0_TILE0=%d\n", TILE[0]' \
        -ex 'printf "CUDA_OXIDE_SHARED_BLOCK0_TILE1=%d\n", TILE[1]' \
        -ex 'printf "CUDA_OXIDE_SHARED_BLOCK0_TILE7=%d\n", TILE[7]' \
        -ex 'printf "CUDA_OXIDE_SHARED_BLOCK0_ADDRESS=%p\n", &TILE' \
        -ex 'cuda block (1,0,0) thread (0,0,0)' \
        -ex 'printf "CUDA_OXIDE_SHARED_BLOCK1_TILE0=%d\n", TILE[0]' \
        -ex 'printf "CUDA_OXIDE_SHARED_BLOCK1_TILE1=%d\n", TILE[1]' \
        -ex 'printf "CUDA_OXIDE_SHARED_BLOCK1_ADDRESS=%p\n", &TILE' \
        -ex 'backtrace' \
        -ex 'kill' \
        "./target/release/$EXAMPLE" ) >"$GDB_LOG" 2>&1
elif [ "$EXAMPLE" = "debug" ]; then
    ( cd "$EXAMPLE_DIR" && timeout 300 "$CUDA_GDB" --batch \
        -ex 'set pagination off' \
        -ex 'set language auto' \
        -ex 'run' \
        -ex 'show language' \
        -ex 'frame 0' \
        -ex 'info args' \
        -ex 'info locals' \
        -ex 'whatis idx_raw' \
        -ex 'printf "CUDA_OXIDE_BREAKPOINT_IDX=%llu\n", idx_raw' \
        -ex 'printf "CUDA_OXIDE_BREAKPOINT_ADDRESS=%p\n", output_elem' \
        -ex 'backtrace' \
        -ex 'continue' \
        --args "./target/release/$EXAMPLE" --breakpoint ) >"$GDB_LOG" 2>&1
else
    ( cd "$EXAMPLE_DIR" && timeout 300 "$CUDA_GDB" --batch \
        -ex 'set pagination off' \
        -ex 'set breakpoint pending on' \
        -ex "break ${BREAK_AT:-test_option}" \
        -ex 'run' \
        -ex 'frame 0' \
        -ex 'info args' \
        -ex 'info locals' \
        -ex 'backtrace' \
        -ex 'kill' \
        "./target/release/$EXAMPLE" ) >"$GDB_LOG" 2>&1
fi

main_gdb_status=$?
if [ "$main_gdb_status" -ne 0 ]; then
    echo "debug-smoketest: FAIL (cuda-gdb exited with status $main_gdb_status; complete log: $GDB_LOG)"
    fail=1
fi

echo "----- cuda-gdb output (tail) -----"
tail -25 "$GDB_LOG"
echo "----------------------------------"

# Verdict: a device breakpoint must have bound and fired, and at least one
# concrete value (scalar, pointer, struct field, or enum payload) must be visible.
if [ "$EXAMPLE" = "debug" ]; then
    grep -q 'received signal SIGTRAP, Trace/breakpoint trap' "$GDB_LOG" \
        || { echo "debug-smoketest: FAIL (debug::breakpoint did not trap)"; fail=1; }
    grep -q 'Switching focus to CUDA kernel' "$GDB_LOG" \
        || { echo "debug-smoketest: FAIL (debug::breakpoint did not stop a CUDA thread)"; fail=1; }
else
    grep -qiE "CUDA thread hit .*Breakpoint" "$GDB_LOG" \
        || { echo "debug-smoketest: FAIL (no device breakpoint hit)"; fail=1; }
fi
grep -qE "= [0-9]|0x[0-9a-f]|\{.*:" "$GDB_LOG"        || { echo "debug-smoketest: FAIL (no inspectable args/locals)"; fail=1; }
grep -qiE "INVALID_PTX|JIT compilation failed|No device code" "$GDB_LOG" && { echo "debug-smoketest: FAIL (PTX did not load under cuda-gdb)"; fail=1; }
grep -Eq '^#0 .*<<<.*>>>' "$GDB_LOG" \
    || { echo "debug-smoketest: FAIL (frame 0 is not a CUDA kernel)"; fail=1; }
grep -q 'No frame at level' "$GDB_LOG" \
    && { echo "debug-smoketest: FAIL (requested debugger frame does not exist)"; fail=1; }

if [ "$EXAMPLE" = "debug_pointer_locals" ]; then
    ptr_address="$(sed -nE 's/^CUDA_OXIDE_PTR_ADDRESS=(0x[[:xdigit:]]+)$/\1/p' "$GDB_LOG" | head -1)"
    fptr_address="$(sed -nE 's/^CUDA_OXIDE_FPTR_ADDRESS=(0x[[:xdigit:]]+)$/\1/p' "$GDB_LOG" | head -1)"
    [ -n "$ptr_address" ] && [ "$ptr_address" != "0x0" ] \
        || { echo "debug-smoketest: FAIL (ptr is not a live non-null address)"; fail=1; }
    [ -n "$fptr_address" ] && [ "$fptr_address" != "0x0" ] \
        || { echo "debug-smoketest: FAIL (fptr is not a live non-null address)"; fail=1; }
    [ -z "$ptr_address" ] || [ -z "$fptr_address" ] || [ "$ptr_address" != "$fptr_address" ] \
        || { echo "debug-smoketest: FAIL (ptr and fptr unexpectedly have the same address)"; fail=1; }
    grep -q '^CUDA_OXIDE_PTR_POINTEE=41$' "$GDB_LOG" \
        || { echo "debug-smoketest: FAIL (*ptr is not the i32 sentinel 41)"; fail=1; }
    grep -Eq '^CUDA_OXIDE_FPTR_POINTEE=13([.]0*)?$' "$GDB_LOG" \
        || { echo "debug-smoketest: FAIL (*fptr is not the f32 sentinel 13.0)"; fail=1; }
    grep -Eq '^CUDA_OXIDE_NULL_PTR=(0x0|\(nil\))$' "$GDB_LOG" \
        || { echo "debug-smoketest: FAIL (null_ptr is not reported as null)"; fail=1; }
    grep -Fq 'The current source language is "auto; currently rust".' "$GDB_LOG" \
        || { echo "debug-smoketest: FAIL (pointer lookup is not using Rust language mode)"; fail=1; }
    # LLVM's exact Rust type names are asserted in verify-debug-info.sh.
    # cuda-gdb currently canonicalizes every NVPTX DW_TAG_pointer_type to
    # `*mut`, including `&T` and `*const T`; the live gate therefore checks
    # pointer shape and the exact pointee without mistaking presentation for
    # producer semantics. Accept the native spelling if a future consumer
    # starts honoring the retained Rust type name.
    grep -A1 '^CUDA_OXIDE_PTR_TYPE$' "$GDB_LOG" \
        | grep -Eq '^type = (&i32|\*const i32|\*mut i32)$' \
        || { echo "debug-smoketest: FAIL (ptr is not a Rust pointer-shaped i32 type)"; fail=1; }
    grep -A1 '^CUDA_OXIDE_FPTR_TYPE$' "$GDB_LOG" \
        | grep -Eq '^type = (&f32|\*const f32|\*mut f32)$' \
        || { echo "debug-smoketest: FAIL (fptr is not a Rust pointer-shaped f32 type)"; fail=1; }
    grep -A1 '^CUDA_OXIDE_NULL_PTR_TYPE$' "$GDB_LOG" \
        | grep -Eq '^type = (&i32|\*const i32|\*mut i32)$' \
        || { echo "debug-smoketest: FAIL (null_ptr is not a Rust pointer-shaped i32 type)"; fail=1; }
elif [ "$EXAMPLE" = "device_global" ]; then
    grep -Fq 'The current source language is "auto; currently rust".' "$GDB_LOG" \
        || { echo "debug-smoketest: FAIL (device-global lookup is not using Rust language mode)"; fail=1; }
    grep -q '^CUDA_OXIDE_DEVICE_COUNTER=1$' "$GDB_LOG" \
        || { echo "debug-smoketest: FAIL (qualified DEVICE_COUNTER lookup/value is wrong)"; fail=1; }
    grep -q '^CUDA_OXIDE_DEVICE_MARKER=12648430$' "$GDB_LOG" \
        || { echo "debug-smoketest: FAIL (qualified DEVICE_MARKER lookup/value is wrong)"; fail=1; }
    grep -q '^CUDA_OXIDE_GLOBAL_LEFT=1$' "$GDB_LOG" \
        || { echo "debug-smoketest: FAIL (qualified debug_left::SAME_LEAF lookup/value is wrong)"; fail=1; }
    grep -q '^CUDA_OXIDE_GLOBAL_RIGHT=2$' "$GDB_LOG" \
        || { echo "debug-smoketest: FAIL (qualified debug_right::SAME_LEAF lookup/value is wrong)"; fail=1; }
    [ "$(grep -c '^type = @global u64$' "$GDB_LOG")" -ge 2 ] \
        || { echo "debug-smoketest: FAIL (AS1 u64 global types are wrong)"; fail=1; }
    [ "$(grep -c '^type = @global u32$' "$GDB_LOG")" -ge 2 ] \
        || { echo "debug-smoketest: FAIL (AS1 u32 global types are wrong)"; fail=1; }
elif [ "$EXAMPLE" = "shared_debug" ]; then
    grep -Fq 'The current source language is "auto; currently rust".' "$GDB_LOG" \
        || { echo "debug-smoketest: FAIL (shared lookup is not using Rust language mode)"; fail=1; }
    [ "$(grep -c '^type = \[i32; 32\]$' "$GDB_LOG")" -ge 2 ] \
        || { echo "debug-smoketest: FAIL (TILE lost its Rust [i32; 32] type)"; fail=1; }
    grep -q '^CUDA_OXIDE_SHARED_SIZE=128$' "$GDB_LOG" \
        || { echo "debug-smoketest: FAIL (TILE size is not 128 bytes)"; fail=1; }
    grep -q '^CUDA_OXIDE_SHARED_BLOCK0_TILE0=0$' "$GDB_LOG" \
        || { echo "debug-smoketest: FAIL (block 0 TILE[0] is wrong)"; fail=1; }
    grep -q '^CUDA_OXIDE_SHARED_BLOCK0_TILE1=1$' "$GDB_LOG" \
        || { echo "debug-smoketest: FAIL (block 0 TILE[1] is wrong)"; fail=1; }
    grep -q '^CUDA_OXIDE_SHARED_BLOCK0_TILE7=7$' "$GDB_LOG" \
        || { echo "debug-smoketest: FAIL (block 0 TILE[7] is wrong)"; fail=1; }
    grep -q '^CUDA_OXIDE_SHARED_BLOCK1_TILE0=100$' "$GDB_LOG" \
        || { echo "debug-smoketest: FAIL (block 1 TILE[0] is wrong)"; fail=1; }
    grep -q '^CUDA_OXIDE_SHARED_BLOCK1_TILE1=101$' "$GDB_LOG" \
        || { echo "debug-smoketest: FAIL (block 1 TILE[1] is wrong)"; fail=1; }
    block0_address="$(sed -nE 's/^CUDA_OXIDE_SHARED_BLOCK0_ADDRESS=(0x[[:xdigit:]]+)$/\1/p' "$GDB_LOG" | head -1)"
    block1_address="$(sed -nE 's/^CUDA_OXIDE_SHARED_BLOCK1_ADDRESS=(0x[[:xdigit:]]+)$/\1/p' "$GDB_LOG" | head -1)"
    [ -n "$block0_address" ] && [ "$block0_address" != "0x0" ] \
        || { echo "debug-smoketest: FAIL (block 0 TILE address is missing/null)"; fail=1; }
    [ "$block0_address" = "$block1_address" ] \
        || { echo "debug-smoketest: FAIL (TILE did not retain its per-block shared offset)"; fail=1; }
    grep -Eq '^#0 .*shared_debug.*<<<.*>>>' "$GDB_LOG" \
        || { echo "debug-smoketest: FAIL (shared breakpoint is not in the kernel frame)"; fail=1; }
elif [ "$EXAMPLE" = "debug" ]; then
    grep -Fq 'The current source language is "auto; currently rust".' "$GDB_LOG" \
        || { echo "debug-smoketest: FAIL (breakpoint lookup is not using Rust language mode)"; fail=1; }
    grep -q '^CUDA_OXIDE_BREAKPOINT_IDX=0$' "$GDB_LOG" \
        || { echo "debug-smoketest: FAIL (debug::breakpoint did not stop lane 0)"; fail=1; }
    grep -Eq '^CUDA_OXIDE_BREAKPOINT_ADDRESS=0x[[:xdigit:]]+$' "$GDB_LOG" \
        || { echo "debug-smoketest: FAIL (breakpoint output pointer is not inspectable)"; fail=1; }
    grep -Eq '^#0 .*breakpoint_test.*<<<.*>>>' "$GDB_LOG" \
        || { echo "debug-smoketest: FAIL (breakpoint kernel frame is missing)"; fail=1; }
    grep -Fq 'PASS breakpoint_test: runtime values = [0, 1, 2, 3, 4, 5, 6, 7]' "$GDB_LOG" \
        || { echo "debug-smoketest: FAIL (breakpoint kernel did not finish after continue)"; fail=1; }
fi

# compiler_features contains dedicated full-debug fixtures. Break on the source
# line immediately after each value is initialized, then require cuda-gdb to
# consume the generated DWARF rather than merely accepting the LLVM metadata.
if [ "$EXAMPLE" = "compiler_features" ]; then
    DEBUG_SOURCE="$EXAMPLE_DIR/src/main.rs"

    # T08/T14 are a paired frame test. At the caller marker `get_mut` has
    # returned, so no helper frame may cover that PC. At the helper's own
    # branch, an honest get_mut -> kernel stack must still be available. Full
    # debug currently achieves both by outlining only DisjointSlice::get_mut;
    # general MIR inlining stays enabled while NVPTX/cuda-gdb cannot faithfully
    # exchange discontiguous inline-scope ranges.
    INLINE_CALLER_MARKER="CUDA_OXIDE_DEBUG_INLINE_CALLER_BREAKPOINT"
    INLINE_CALLER_LINE="$(grep -nF "$INLINE_CALLER_MARKER" "$DEBUG_SOURCE" | head -1 | cut -d: -f1)"
    DISJOINT_SOURCE="$REPO_ROOT/crates/cuda-device/src/disjoint.rs"
    INLINE_HELPER_LINE="$(grep -nF 'if size_of::<T>() != 0 && idx.is_valid() && i < self.len {' "$DISJOINT_SOURCE" | head -1 | cut -d: -f1)"

    if [ -z "$INLINE_CALLER_LINE" ] || [ -z "$INLINE_HELPER_LINE" ]; then
        echo "debug-smoketest: FAIL (inline caller/helper breakpoint source not found)"
        fail=1
    else
        ( cd "$EXAMPLE_DIR" && timeout 300 "$CUDA_GDB" --batch \
            -ex 'set pagination off' \
            -ex 'set language auto' \
            -ex 'set breakpoint pending on' \
            -ex "break $DEBUG_SOURCE:$INLINE_CALLER_LINE" \
            -ex 'run' \
            -ex 'frame 0' \
            -ex 'print idx' \
            -ex 'print out_elem' \
            -ex 'backtrace' \
            -ex 'kill' \
            "./target/release/$EXAMPLE" ) >"$INLINE_CALLER_GDB_LOG" 2>&1

        inline_caller_status=$?
        if [ "$inline_caller_status" -ne 0 ]; then
            echo "debug-smoketest: FAIL (inline-caller cuda-gdb exited with status $inline_caller_status; complete log: $INLINE_CALLER_GDB_LOG)"
            fail=1
        fi

        echo "----- cuda-gdb inline caller output (tail) -----"
        tail -30 "$INLINE_CALLER_GDB_LOG"
        echo "------------------------------------------------"

        grep -qiE "CUDA thread hit .*Breakpoint" "$INLINE_CALLER_GDB_LOG" \
            || { echo "debug-smoketest: FAIL (inline caller breakpoint did not hit)"; fail=1; }
        grep -Eq '^#0 .*test_option.*<<<.*>>>' "$INLINE_CALLER_GDB_LOG" \
            || { echo "debug-smoketest: FAIL (caller PC is not in the test_option kernel frame)"; fail=1; }
        ! grep -Eq '^#[0-9]+ .*get_mut' "$INLINE_CALLER_GDB_LOG" \
            || { echo "debug-smoketest: FAIL (T08: get_mut incorrectly covers caller-only PC)"; fail=1; }
        grep -qE '\$[0-9]+ = ThreadIndex \{raw: 0\}' "$INLINE_CALLER_GDB_LOG" \
            || { echo "debug-smoketest: FAIL (caller idx is not inspectable)"; fail=1; }
        inline_out_address="$(sed -nE 's/^\$[0-9]+ = (\([^)]*\) )?(0x[[:xdigit:]]+)$/\2/p' "$INLINE_CALLER_GDB_LOG" | head -1)"
        [ -n "$inline_out_address" ] && [ "$inline_out_address" != "0x0" ] \
            || { echo "debug-smoketest: FAIL (caller out_elem is missing/null)"; fail=1; }
        grep -qiE "INVALID_PTX|JIT compilation failed|No device code" "$INLINE_CALLER_GDB_LOG" \
            && { echo "debug-smoketest: FAIL (inline-caller PTX did not load)"; fail=1; }

        ( cd "$EXAMPLE_DIR" && timeout 300 "$CUDA_GDB" --batch \
            -ex 'set pagination off' \
            -ex 'set language auto' \
            -ex 'set breakpoint pending on' \
            -ex 'break test_option' \
            -ex 'run' \
            -ex 'delete 1' \
            -ex "break $DISJOINT_SOURCE:$INLINE_HELPER_LINE" \
            -ex 'continue' \
            -ex 'frame 0' \
            -ex 'info args' \
            -ex 'backtrace' \
            -ex 'kill' \
            "./target/release/$EXAMPLE" ) >"$INLINE_HELPER_GDB_LOG" 2>&1

        inline_helper_status=$?
        if [ "$inline_helper_status" -ne 0 ]; then
            echo "debug-smoketest: FAIL (inline-helper cuda-gdb exited with status $inline_helper_status; complete log: $INLINE_HELPER_GDB_LOG)"
            fail=1
        fi

        echo "----- cuda-gdb helper-frame output (tail) -----"
        tail -30 "$INLINE_HELPER_GDB_LOG"
        echo "-----------------------------------------------"

        grep -qiE "CUDA thread hit .*Breakpoint" "$INLINE_HELPER_GDB_LOG" \
            || { echo "debug-smoketest: FAIL (helper source breakpoint did not hit)"; fail=1; }
        grep -Eq '^#0 .*get_mut' "$INLINE_HELPER_GDB_LOG" \
            || { echo "debug-smoketest: FAIL (T14: get_mut helper frame is missing)"; fail=1; }
        grep -Eq '^#1 .*test_option.*<<<.*>>>' "$INLINE_HELPER_GDB_LOG" \
            || { echo "debug-smoketest: FAIL (T14: helper has no test_option kernel caller frame)"; fail=1; }
        grep -q 'idx = ThreadIndex {raw: 0}' "$INLINE_HELPER_GDB_LOG" \
            || { echo "debug-smoketest: FAIL (T14: helper idx is not inspectable)"; fail=1; }
        helper_self_address="$(sed -nE 's/^self = (0x[[:xdigit:]]+)$/\1/p' "$INLINE_HELPER_GDB_LOG" | head -1)"
        [ -n "$helper_self_address" ] && [ "$helper_self_address" != "0x0" ] \
            || { echo "debug-smoketest: FAIL (T14: helper self is missing/null)"; fail=1; }
        grep -qiE "INVALID_PTX|JIT compilation failed|No device code" "$INLINE_HELPER_GDB_LOG" \
            && { echo "debug-smoketest: FAIL (helper-frame PTX did not load)"; fail=1; }
    fi

    CLOSURE_MARKER="CUDA_OXIDE_DEBUG_CLOSURE_BREAKPOINT"
    CLOSURE_LINE="$(grep -nF "$CLOSURE_MARKER" "$DEBUG_SOURCE" | head -1 | cut -d: -f1)"

    if [ -z "$CLOSURE_LINE" ]; then
        echo "debug-smoketest: FAIL (closure debug breakpoint marker not found)"
        fail=1
    else
        ( cd "$EXAMPLE_DIR" && timeout 300 "$CUDA_GDB" --batch \
            -ex 'set pagination off' \
            -ex 'set breakpoint pending on' \
            -ex "break $DEBUG_SOURCE:$CLOSURE_LINE" \
            -ex 'run' \
            -ex 'frame 0' \
            -ex 'ptype closure' \
            -ex 'print closure' \
            -ex 'backtrace' \
            -ex 'kill' \
            "./target/release/$EXAMPLE" ) >"$CLOSURE_GDB_LOG" 2>&1

        closure_gdb_status=$?
        if [ "$closure_gdb_status" -ne 0 ]; then
            echo "debug-smoketest: FAIL (closure cuda-gdb exited with status $closure_gdb_status; complete log: $CLOSURE_GDB_LOG)"
            fail=1
        fi

        echo "----- cuda-gdb closure output (tail) -----"
        tail -25 "$CLOSURE_GDB_LOG"
        echo "------------------------------------------"

        grep -qiE "CUDA thread hit .*Breakpoint" "$CLOSURE_GDB_LOG" || { echo "debug-smoketest: FAIL (closure source breakpoint did not hit)"; fail=1; }
        grep -qE "capture_0[[:space:]]*:[[:space:]]*17([,}]|$)" "$CLOSURE_GDB_LOG" || { echo "debug-smoketest: FAIL (closure capture_0 is not the sentinel 17)"; fail=1; }
        grep -qE "capture_1[[:space:]]*:[[:space:]]*4294967328([,}]|$)" "$CLOSURE_GDB_LOG" || { echo "debug-smoketest: FAIL (closure capture_1 is not the sentinel 4294967328)"; fail=1; }
        grep -qiE "INVALID_PTX|JIT compilation failed|No device code" "$CLOSURE_GDB_LOG" && { echo "debug-smoketest: FAIL (closure-debug PTX did not load under cuda-gdb)"; fail=1; }
    fi

    ENUM_MARKER="CUDA_OXIDE_DEBUG_ENUM_BREAKPOINT"
    ENUM_LINE="$(grep -nF "$ENUM_MARKER" "$DEBUG_SOURCE" | head -1 | cut -d: -f1)"

    if [ -z "$ENUM_LINE" ]; then
        echo "debug-smoketest: FAIL (enum debug breakpoint marker not found)"
        fail=1
    else
        ( cd "$EXAMPLE_DIR" && timeout 300 "$CUDA_GDB" --batch \
            -ex 'set pagination off' \
            -ex 'set breakpoint pending on' \
            -ex "break $DEBUG_SOURCE:$ENUM_LINE" \
            -ex 'run' \
            -ex 'frame 0' \
            -ex 'ptype option_value' \
            -ex 'print option_value' \
            -ex 'ptype result_value' \
            -ex 'print result_value' \
            -ex 'ptype direct_value' \
            -ex 'print direct_value' \
            -ex 'ptype niche_value' \
            -ex 'print niche_value' \
            -ex 'backtrace' \
            -ex 'kill' \
            "./target/release/$EXAMPLE" ) >"$ENUM_GDB_LOG" 2>&1

        enum_gdb_status=$?
        if [ "$enum_gdb_status" -ne 0 ]; then
            echo "debug-smoketest: FAIL (enum cuda-gdb exited with status $enum_gdb_status; complete log: $ENUM_GDB_LOG)"
            fail=1
        fi

        echo "----- cuda-gdb enum output (tail) -----"
        tail -45 "$ENUM_GDB_LOG"
        echo "---------------------------------------"

        grep -qiE "CUDA thread hit .*Breakpoint" "$ENUM_GDB_LOG" || { echo "debug-smoketest: FAIL (enum source breakpoint did not hit)"; fail=1; }
        grep -qE '\$[0-9]+ = ([^[:space:]]+::)?Some \(8\)' "$ENUM_GDB_LOG" || { echo "debug-smoketest: FAIL (Option<u32> active variant/payload is not inspectable)"; fail=1; }
        grep -qE '\$[0-9]+ = ([^[:space:]]+::)?Err \(4294967305\)' "$ENUM_GDB_LOG" || { echo "debug-smoketest: FAIL (Result<u32,u64> active Err variant/payload is not inspectable)"; fail=1; }
        grep -qE '\$[0-9]+ = ([^[:space:]]+::)?Wide \(8589934603\)' "$ENUM_GDB_LOG" || { echo "debug-smoketest: FAIL (direct-tag custom enum active variant/payload is not inspectable)"; fail=1; }
        grep -qE '\$[0-9]+ = ([^[:space:]]+::)?Some \(0x[[:xdigit:]]+\)' "$ENUM_GDB_LOG" || { echo "debug-smoketest: FAIL (niche Option<&u32> active variant/payload is not inspectable)"; fail=1; }
        grep -qF '<No data fields>' "$ENUM_GDB_LOG" && { echo "debug-smoketest: FAIL (enum value resolved without an active variant payload)"; fail=1; }
        grep -qiE "INVALID_PTX|JIT compilation failed|No device code" "$ENUM_GDB_LOG" && { echo "debug-smoketest: FAIL (enum-debug PTX did not load under cuda-gdb)"; fail=1; }
    fi

    ENUM_PROJECTION_MARKER="CUDA_OXIDE_DEBUG_ENUM_PROJECTION_BREAKPOINT"
    ENUM_PROJECTION_LINE="$(grep -nF "$ENUM_PROJECTION_MARKER" "$DEBUG_SOURCE" | head -1 | cut -d: -f1)"

    if [ -z "$ENUM_PROJECTION_LINE" ]; then
        echo "debug-smoketest: FAIL (enum projection debug breakpoint marker not found)"
        fail=1
    else
        ( cd "$EXAMPLE_DIR" && timeout 300 "$CUDA_GDB" --batch \
            -ex 'set pagination off' \
            -ex 'set breakpoint pending on' \
            -ex "break $DEBUG_SOURCE:$ENUM_PROJECTION_LINE" \
            -ex 'run' \
            -ex 'frame 0' \
            -ex 'ptype projected_enum_payload' \
            -ex 'print projected_enum_payload' \
            -ex 'backtrace' \
            -ex 'kill' \
            "./target/release/$EXAMPLE" ) >"$ENUM_PROJECTION_GDB_LOG" 2>&1

        enum_projection_gdb_status=$?
        if [ "$enum_projection_gdb_status" -ne 0 ]; then
            echo "debug-smoketest: FAIL (enum-projection cuda-gdb exited with status $enum_projection_gdb_status; complete log: $ENUM_PROJECTION_GDB_LOG)"
            fail=1
        fi

        echo "----- cuda-gdb enum projection output (tail) -----"
        tail -25 "$ENUM_PROJECTION_GDB_LOG"
        echo "--------------------------------------------------"

        grep -qiE "CUDA thread hit .*Breakpoint" "$ENUM_PROJECTION_GDB_LOG" || { echo "debug-smoketest: FAIL (enum projection source breakpoint did not hit)"; fail=1; }
        grep -qE '\$[0-9]+ = 8589934603([[:space:]]|$)' "$ENUM_PROJECTION_GDB_LOG" || { echo "debug-smoketest: FAIL (enum Downcast -> Field payload binding is not inspectable)"; fail=1; }
        grep -qiE "INVALID_PTX|JIT compilation failed|No device code" "$ENUM_PROJECTION_GDB_LOG" && { echo "debug-smoketest: FAIL (enum-projection debug PTX did not load under cuda-gdb)"; fail=1; }
    fi

    PROJECTION_MARKER="CUDA_OXIDE_DEBUG_PROJECTION_BREAKPOINT"
    PROJECTION_LINE="$(grep -nF "$PROJECTION_MARKER" "$DEBUG_SOURCE" | head -1 | cut -d: -f1)"

    if [ -z "$PROJECTION_LINE" ]; then
        echo "debug-smoketest: FAIL (projection debug breakpoint marker not found)"
        fail=1
    else
        ( cd "$EXAMPLE_DIR" && timeout 300 "$CUDA_GDB" --batch \
            -ex 'set pagination off' \
            -ex 'set breakpoint pending on' \
            -ex "break $DEBUG_SOURCE:$PROJECTION_LINE" \
            -ex 'run' \
            -ex 'frame 0' \
            -ex 'ptype projected_field' \
            -ex 'print projected_field' \
            -ex 'ptype projected_tuple' \
            -ex 'print projected_tuple' \
            -ex 'ptype projected_array' \
            -ex 'print projected_array' \
            -ex 'backtrace' \
            -ex 'kill' \
            "./target/release/$EXAMPLE" ) >"$PROJECTION_GDB_LOG" 2>&1

        projection_gdb_status=$?
        if [ "$projection_gdb_status" -ne 0 ]; then
            echo "debug-smoketest: FAIL (projection cuda-gdb exited with status $projection_gdb_status; complete log: $PROJECTION_GDB_LOG)"
            fail=1
        fi

        echo "----- cuda-gdb projection output (tail) -----"
        tail -35 "$PROJECTION_GDB_LOG"
        echo "---------------------------------------------"

        grep -qiE "CUDA thread hit .*Breakpoint" "$PROJECTION_GDB_LOG" || { echo "debug-smoketest: FAIL (projection source breakpoint did not hit)"; fail=1; }
        grep -qE '\$[0-9]+ = 18([[:space:]]|$)' "$PROJECTION_GDB_LOG" || { echo "debug-smoketest: FAIL (struct.field projection is not inspectable)"; fail=1; }
        grep -qE '\$[0-9]+ = 4294967329([[:space:]]|$)' "$PROJECTION_GDB_LOG" || { echo "debug-smoketest: FAIL (tuple.1 projection is not inspectable)"; fail=1; }
        grep -qE '\$[0-9]+ = 37([[:space:]]|$)' "$PROJECTION_GDB_LOG" || { echo "debug-smoketest: FAIL (array constant-index projection is not inspectable)"; fail=1; }
        grep -qiE "INVALID_PTX|JIT compilation failed|No device code" "$PROJECTION_GDB_LOG" && { echo "debug-smoketest: FAIL (projection-debug PTX did not load under cuda-gdb)"; fail=1; }
    fi

    RUNTIME_INDEX_MARKER="CUDA_OXIDE_DEBUG_RUNTIME_INDEX_BREAKPOINT"
    RUNTIME_INDEX_LINE="$(grep -nF "$RUNTIME_INDEX_MARKER" "$DEBUG_SOURCE" | head -1 | cut -d: -f1)"

    if [ -z "$RUNTIME_INDEX_LINE" ]; then
        echo "debug-smoketest: FAIL (runtime-index debug breakpoint marker not found)"
        fail=1
    else
        ( cd "$EXAMPLE_DIR" && timeout 300 "$CUDA_GDB" --batch \
            -ex 'set pagination off' \
            -ex 'set breakpoint pending on' \
            -ex "break $DEBUG_SOURCE:$RUNTIME_INDEX_LINE" \
            -ex 'run' \
            -ex 'frame 0' \
            -ex 'ptype projected_runtime' \
            -ex 'print projected_runtime' \
            -ex 'print *projected_runtime' \
            -ex 'backtrace' \
            -ex 'kill' \
            "./target/release/$EXAMPLE" ) >"$RUNTIME_INDEX_GDB_LOG" 2>&1

        runtime_index_gdb_status=$?
        if [ "$runtime_index_gdb_status" -ne 0 ]; then
            echo "debug-smoketest: FAIL (runtime-index cuda-gdb exited with status $runtime_index_gdb_status; complete log: $RUNTIME_INDEX_GDB_LOG)"
            fail=1
        fi

        echo "----- cuda-gdb runtime-index output (tail) -----"
        tail -30 "$RUNTIME_INDEX_GDB_LOG"
        echo "------------------------------------------------"

        grep -qiE "CUDA thread hit .*Breakpoint" "$RUNTIME_INDEX_GDB_LOG" || { echo "debug-smoketest: FAIL (runtime-index source breakpoint did not hit)"; fail=1; }
        grep -qE '\$[0-9]+ = 55([[:space:]]|$)' "$RUNTIME_INDEX_GDB_LOG" || { echo "debug-smoketest: FAIL (runtime-index fixed-array reference is not inspectable)"; fail=1; }
        grep -qiE 'optimized out|No symbol.*projected_runtime|not available' "$RUNTIME_INDEX_GDB_LOG" && { echo "debug-smoketest: FAIL (runtime-index binding is unavailable)"; fail=1; }
        grep -qiE "INVALID_PTX|JIT compilation failed|No device code" "$RUNTIME_INDEX_GDB_LOG" && { echo "debug-smoketest: FAIL (runtime-index debug PTX did not load under cuda-gdb)"; fail=1; }
    fi

    DEREF_MARKER="CUDA_OXIDE_DEBUG_DEREF_BREAKPOINT"
    DEREF_LINE="$(grep -nF "$DEREF_MARKER" "$DEBUG_SOURCE" | head -1 | cut -d: -f1)"

    if [ -z "$DEREF_LINE" ]; then
        echo "debug-smoketest: FAIL (dereference debug breakpoint marker not found)"
        fail=1
    else
        ( cd "$EXAMPLE_DIR" && timeout 300 "$CUDA_GDB" --batch \
            -ex 'set pagination off' \
            -ex 'set breakpoint pending on' \
            -ex "break $DEBUG_SOURCE:$DEREF_LINE" \
            -ex 'run' \
            -ex 'frame 0' \
            -ex 'ptype deref_field' \
            -ex 'print deref_field' \
            -ex 'ptype deref_value' \
            -ex 'print deref_value' \
            -ex 'backtrace' \
            -ex 'kill' \
            "./target/release/$EXAMPLE" ) >"$DEREF_GDB_LOG" 2>&1

        deref_gdb_status=$?
        if [ "$deref_gdb_status" -ne 0 ]; then
            echo "debug-smoketest: FAIL (dereference cuda-gdb exited with status $deref_gdb_status; complete log: $DEREF_GDB_LOG)"
            fail=1
        fi

        echo "----- cuda-gdb dereference projection output (tail) -----"
        tail -30 "$DEREF_GDB_LOG"
        echo "----------------------------------------------------------"

        grep -qiE "CUDA thread hit .*Breakpoint" "$DEREF_GDB_LOG" || { echo "debug-smoketest: FAIL (dereference projection source breakpoint did not hit)"; fail=1; }
        grep -qE '\$[0-9]+ = 18([[:space:]]|$)' "$DEREF_GDB_LOG" || { echo "debug-smoketest: FAIL (dereference field projection is not inspectable)"; fail=1; }
        grep -qE '\$[0-9]+ = 41([[:space:]]|$)' "$DEREF_GDB_LOG" || { echo "debug-smoketest: FAIL (dereference projection is not inspectable)"; fail=1; }
        grep -qiE "INVALID_PTX|JIT compilation failed|No device code" "$DEREF_GDB_LOG" && { echo "debug-smoketest: FAIL (dereference-debug PTX did not load under cuda-gdb)"; fail=1; }
    fi
fi

if [ "$fail" -eq 0 ]; then
    if [ "$EXAMPLE" = "compiler_features" ]; then
        echo "debug-smoketest: PASS (source debugging, caller/helper frames, closure environments, Rust enums, and static/runtime-index/enum/dereference projections verified on $ARCH)"
    elif [ "$EXAMPLE" = "debug_pointer_locals" ]; then
        echo "debug-smoketest: PASS (raw-pointer debug storage, values, pointees, types, and null control verified on $ARCH)"
    elif [ "$EXAMPLE" = "device_global" ]; then
        echo "debug-smoketest: PASS (AS1 shape and qualified Rust-mode global lookup verified on $ARCH)"
    elif [ "$EXAMPLE" = "shared_debug" ]; then
        echo "debug-smoketest: PASS (AS3 class 8, Rust lookup, type, per-block values, and shared offset verified on $ARCH)"
    elif [ "$EXAMPLE" = "debug" ]; then
        echo "debug-smoketest: PASS (line-table shape and debug::breakpoint trap, locals, frame, and continue verified on $ARCH)"
    else
        echo "debug-smoketest: PASS (source debugging + info args/locals verified on $ARCH)"
    fi
    exit 0
fi
exit 1
