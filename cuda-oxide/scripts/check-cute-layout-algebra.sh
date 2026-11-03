#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Enforce the cute-rs hard rules on (1) `cute-layout` single source of truth,
# (2) CuTe layout algebra, and (3) Rust type encoding of TMA, pipelines, MMA,
# TMEM, barriers, and partitions in `*_cute` kernels:
#   1. Single source of truth in `cute-layout`: never reinvent XOR swizzle bit
#      math (`^ ((... >> ...) << ...)` or `^ ((... & ...) >> ...)`), string-
#      interpolated layout construction (`format!(...).parse()`), or layout
#      definitions (`struct SharedLayout` / `trait SmemLayout`) outside
#      `cute-layout`.
#   2. Rust type encoding in `*_cute` kernels: TMA transfers, pipelines, MMA
#      operations, TMEM operations, barriers, and partitions must go through
#      `cute-rs` typed abstractions—never bypass them in `*_cute` kernels via
#      direct `cuda_device::*` or raw `ctm_rs::{tma, mma, tcgen05, async_copy,
#      barrier, cluster, fence, groups, convert, float}` / raw `mbarrier_*` or
#      `cp_async_bulk_tensor_*` instruction calls.
#   3. No hand-rolled layout algebra in `*_cute` kernels: never write manual
#      tile-swizzle helpers (`fn swizzle_tile`), manual thread-layout bit
#      unpacking (`tid & 63`, `tid >> 6`), manual TMEM row bit-shifts
#      (`(warp... * 32) << 16`), or raw 2D/3D global pointer offset formulas
#      (`.add((... * ... + ...) * ...)`) in `*_cute` kernels.
#
# Temporary rough work is allowed while developing and verifying correctness,
# but the final implementation must use `cute-layout` / `cute-rs` exclusively
# so this guard passes cleanly.
set -euo pipefail

export LC_ALL=C

cd "$(dirname "$0")/.."
GIT_ROOT="$(git rev-parse --show-toplevel)"
cd "${GIT_ROOT}"

if ! command -v grep >/dev/null 2>&1; then
    echo "error: grep is required to verify cute-layout algebra usage" >&2
    exit 1
fi

swizzle_xor_pattern='\^[[:space:]]*\(\('
format_parse_pattern='format!\("(\(\(|\{shape\}:\{stride\}|\(\{rows\},\{columns\}\))'
layout_def_outside_pattern='pub[[:space:]]+(struct[[:space:]]+(SharedLayout|RowMajor|ColMajor|ComposedLayout)|trait[[:space:]]+(SmemLayout|StaticLayout|ReifySmem2D))\b'
kernel_hand_math_pattern='(fn[[:space:]]+swizzle_tile\b|\([[:space:]]*tid[[:space:]]*(&|>>)[[:space:]]*6[34]?\)|\([[:space:]]*warp[_a-z0-9]*[[:space:]]*\*[[:space:]]*32[[:space:]]*\)[[:space:]]*<<[[:space:]]*16|\b(q_out|k_cache|v_cache)\.add\()'
kernel_untyped_op_pattern='(\bcuda_device::|\bctm_rs::(tma|mma|tcgen05|async_copy|barrier|cluster|fence|groups|convert|float)\b|\bcp_async_bulk_tensor_|\bmbarrier_(init|try_wait|arrive)\b|\btcgen05_(mma|ld|st|cp|alloc|dealloc|wait|fence|commit)\b)'

# Canary self-test: prove each regex matches a known violation before scanning.
canary="$(mktemp -d)"
trap 'rm -rf "${canary}"' EXIT
cat >"${canary}/canary.rs" <<'EOF'
pub struct SharedLayout<const R: u32, const C: u32>;
fn swizzle_tile(tile: u32, mtiles: u32, ntiles: u32) -> (u32, u32) { (tile, mtiles + ntiles) }
fn bad_swizzle(plain: usize, mask: usize) -> usize { plain ^ (((plain & mask) >> 3) << 0) }
fn bad_format(shape: &str, stride: &str) -> String { format!("{shape}:{stride}") }
fn bad_tid(tid: u32) -> (u32, u32) { ((tid & 63), (tid >> 6)) }
fn bad_tmem(tmem: u32, warp_id: u32) -> u32 { tmem + ((warp_id * 32) << 16) }
unsafe fn bad_ptr(q_out: *mut u8, idx: usize) -> *mut u8 { q_out.add(idx) }
fn bad_bypass() { ctm_rs::tma::cp_async_bulk_tensor_shared_cta_global(); }
EOF

for pat in \
    "${swizzle_xor_pattern}" \
    "${format_parse_pattern}" \
    "${layout_def_outside_pattern}" \
    "${kernel_hand_math_pattern}" \
    "${kernel_untyped_op_pattern}"; do
    if ! grep -En "${pat}" "${canary}/canary.rs" >/dev/null; then
        echo "error: check-cute-layout-algebra self-test failed for pattern: ${pat}" >&2
        exit 1
    fi
done

status=0

# 1. Check for reinvented XOR swizzle math or layout definitions outside cute-layout.
set +e
swizzle_hits="$(grep -rEn --include='*.rs' "${swizzle_xor_pattern}" \
    cuda-oxide/cute-rs-preview/cute-rs/src \
    cuda-oxide/cute-rs-preview/dialect-cute/src |
    grep -vE ':[0-9]+:[[:space:]]*//' || true)"
layout_def_hits="$(grep -rEn --include='*.rs' "${layout_def_outside_pattern}" \
    cuda-oxide/cute-rs-preview/cute-rs/src \
    cuda-oxide/cute-rs-preview/dialect-cute/src |
    grep -vE ':[0-9]+:[[:space:]]*//' || true)"
set -e

if [ -n "${swizzle_hits}" ]; then
    status=1
    echo "error: reinvented XOR swizzle bit math found outside cute-layout (use cute_layout::Swizzle::apply):" >&2
    printf '%s\n' "${swizzle_hits}" | sed 's/^/  /' >&2
    echo >&2
fi

if [ -n "${layout_def_hits}" ]; then
    status=1
    echo "error: layout struct/trait defined outside cute-layout (move to cute-layout and re-export):" >&2
    printf '%s\n' "${layout_def_hits}" | sed 's/^/  /' >&2
    echo >&2
fi

# 2. Check for string-formatted Layout construction (`format!(...).parse()`) in
#    non-test library code (use `cute-layout` constructors and algebra instead).
set +e
format_hits="$(grep -rEn --include='*.rs' "${format_parse_pattern}" \
    cuda-oxide/cute-rs-preview/cute-rs/src \
    cuda-oxide/cute-rs-preview/dialect-cute/src |
    grep -v 'tests\.rs:' |
    grep -vE ':[0-9]+:[[:space:]]*//' || true)"
set -e

if [ -n "${format_hits}" ]; then
    status=1
    echo "error: string-interpolated layout construction found outside cute-layout (use cute-layout algebra/constructors):" >&2
    printf '%s\n' "${format_hits}" | sed 's/^/  /' >&2
    echo >&2
fi

# 3. Check `*_cute` example and model kernels for:
#    (a) hand-rolled tile swizzling, thread coordinate unpacking, TMEM row shifts,
#        or raw global output pointer indexing; and
#    (b) bypassing `cute-rs` type-encoded TMA, pipeline, MMA, TMEM, or barrier
#        abstractions via raw `cuda_device` / `ctm_rs` instruction calls.
mapfile -t cute_kernel_dirs < <(find \
    cuda-oxide/cute-rs-preview/examples \
    cuda-oxide/cute-rs-preview/models \
    -maxdepth 2 -type d -name 'src' | grep '_cute/src$' | sort)

set +e
kernel_hits="$(grep -rEn --include='*.rs' "${kernel_hand_math_pattern}" \
    "${cute_kernel_dirs[@]}" |
    grep -vE ':[0-9]+:[[:space:]]*//' || true)"
untyped_hits="$(grep -rEn --include='*.rs' "${kernel_untyped_op_pattern}" \
    "${cute_kernel_dirs[@]}" |
    grep -v 'cute_boundary\.rs:' |
    grep -vE ':[0-9]+:[[:space:]]*//' || true)"
set -e

if [ -n "${kernel_hits}" ]; then
    status=1
    echo "error: hand-rolled layout/coordinate/TMEM math found in *_cute kernels (use cute-layout / cute-rs algebra):" >&2
    printf '%s\n' "${kernel_hits}" | sed 's/^/  /' >&2
    echo >&2
fi

if [ -n "${untyped_hits}" ]; then
    status=1
    echo "error: untyped instruction-level TMA/pipeline/MMA/TMEM call found in *_cute kernel (encode in cute-rs types):" >&2
    printf '%s\n' "${untyped_hits}" | sed 's/^/  /' >&2
    echo >&2
fi

if [ "${status}" -ne 0 ]; then
    echo "hint: all layout, swizzle, tile-rasterization, and coordinate mappings must use" >&2
    echo "      cute-layout / cute-rs layout algebra, and all TMA, pipeline, MMA, TMEM," >&2
    echo "      and partition operations must be encoded in cute-rs Rust types." >&2
    echo "      Temporary rough work is allowed during development only and must be" >&2
    echo "      replaced before finishing." >&2
    exit 1
fi

echo "OK: cute-layout single source of truth, layout algebra, and cute-rs type-encoding guards passed."
