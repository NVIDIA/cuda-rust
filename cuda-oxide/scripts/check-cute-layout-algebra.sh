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
#      direct `cuda_device::*` or raw `cutlass_primitives::{tma, mma, tcgen05, async_copy,
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
kernel_untyped_op_pattern='(\bcuda_device::|\b(cutlass_primitives|prims|ctm_rs)::(tma|mma|tcgen05|async_copy|barrier|cluster|fence|groups|convert|float)\b|\bcp_async_bulk_tensor_|\bmbarrier_(init|try_wait|arrive)\b|\btcgen05_(mma|ld|st|cp|alloc|dealloc|wait|fence|commit)\b)'
moe192_raw_cast_pattern='(\.(as_ptr|as_mut_ptr)\(\)\.(cast::<|add\()|\.cast::<|\.add\()'

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
fn bad_bypass() { cutlass_primitives::tma::cp_async_bulk_tensor_shared_cta_global(); }
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
#        abstractions via raw `cuda_device` / `cutlass_primitives` instruction calls.
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
moe192_cast_hits="$(grep -En "${moe192_raw_cast_pattern}" \
    cuda-oxide/cute-rs-preview/models/nemotron3_moe_cute/src/main.rs \
    cuda-oxide/cute-rs-preview/models/nemotron3_moe_cute/src/moe_192e.rs |
    grep -vE ':[0-9]+:[[:space:]]*//' || true)"
set -e

if [ -n "${moe192_cast_hits}" ]; then
    status=1
    echo "error: raw .as_ptr().cast::<...> found in moe_192e.rs kernel body (use typed StridedTensor / StridedTensorMut / slice views):" >&2
    printf '%s\n' "${moe192_cast_hits}" | sed 's/^/  /' >&2
    echo >&2
fi

# 4. Direct CuTeDSL Python AST symbol & purity check across `cute-rs/src/` and `nemotron3_moe_cute`.
if command -v python3 >/dev/null 2>&1; then
    python3 - <<'PY' || status=1
import ast
import glob
import os
import re
import sys

cutedsl_candidates = [
    "cuda-oxide/cute-rs-preview/.venv/lib/python3.12/site-packages/nvidia_cutlass_dsl/dsl_packages/cutlass",
    os.path.expanduser("~/cudagym-env/lib/python3.12/site-packages/nvidia_cutlass_dsl/dsl_packages/cutlass"),
]
cutedsl_root = next((p for p in cutedsl_candidates if os.path.isdir(p)), None)

cutedsl_symbols = set()
if cutedsl_root is not None:
    for sub in ("cute", "pipeline", "utils"):
        for path in glob.glob(os.path.join(cutedsl_root, sub, "**", "*.py"), recursive=True):
            with open(path, encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=path)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if not node.name.startswith("_"):
                        cutedsl_symbols.add(node.name)
                elif isinstance(node, ast.Assign):
                    for t in node.targets:
                        if isinstance(t, ast.Name) and t.id == "__all__" and isinstance(node.value, (ast.List, ast.Tuple)):
                            for elt in node.value.elts:
                                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                    cutedsl_symbols.add(elt.value)
    if len(cutedsl_symbols) < 500:
        sys.exit(f"error: CuTeDSL AST extraction self-test failed ({len(cutedsl_symbols)} symbols)")

# Documented Rust constructors/idioms from cuda-oxide-book/cute/cutedsl-comparison.md
rust_idiom_allowlist = {
    "new", "new_with_lbo", "from_raw", "from_raw_bytes", "from_stage_bytes",
    "shared_offset", "as_ptr", "tile_unchecked", "tmem_row_address",
    "slice_address", "store_async", "sync_cluster", "alloc", "dealloc",
    "load", "load_deferred", "store", "store_deferred", "store_packed",
    "store_packed_deferred", "store_packed_8", "create_tensor_map_tiled",
    "init", "producer_expect_tx", "acquire", "wait", "full_at", "empty_at",
    "full_ptr", "mma_full", "mma_empty", "thread_full", "thread_empty",
    "full_at_cursor", "empty_at_cursor", "acquire_cursor", "wait_cursor",
    "producer_expect_tx_cursor", "mma_full_cursor", "mma_empty_cursor",
    "thread_full_cursor", "thread_empty_cursor", "commit_threads",
    "release_threads", "commit_mma", "release_mma", "to_tensor_ssa",
    # Documented Tensor / TensorSSA / StridedTensor methods in cute-rs/src/tensor/
    "len", "is_empty", "tile_count", "base", "is_full", "from_array",
    "as_array", "into_array", "splat", "from_fn", "get", "set", "map",
    "zip_map", "fold", "gather", "to", "bitcast", "reduce", "abs", "fmax",
    "concat", "convert_tile", "bitcast_tile", "from_storage", "storage",
    "layout", "from_slice", "from_slice_as", "from_mut_slice",
    "from_disjoint_slice", "from_disjoint_slice_as", "from_raw_parts",
    "from_shared_bytes", "store_at", "write_at", "update_at", "at_unchecked",
    "get_unchecked", "ptr_at_unchecked", "ptr_at_2d_unchecked",
    "offset_unchecked", "subview", "subview_unchecked", "reborrow",
    "load_unchecked", "load_strided_unchecked", "load_f32x4",
    "store_unchecked", "store_strided_unchecked",
    "store_global_no_allocate_at", "store_global_no_allocate_words_at",
    "atomic_add_bf16x2_at_2d", "crd2idx", "where_", "slice", "valid_len",
    "map_to", "zip_map_to", "from_bits", "to_bits",
}

pub_fn_re = re.compile(r"^\s*pub\s+(?:const\s+)?(?:unsafe\s+)?fn\s+([a-zA-Z0-9_]+)")
forbidden_cute_rs_re = re.compile(
    r"\b(route_histogram|route_and_load|store_routed_token|unload_relu2|unload_dual_m|unload_triple_m|copy_scales_and_gemm|copy_b_scales_and_gemm|load_stationary_weights|stage_page_index|wait_and_read_page_index|read_page_index_and_release|rms_inv_std|quantize_and_store_e4m3|partition_pipeline_barriers|partition_triple_m_barriers|to_u128|concat8|reduce_add|reduce_max)\b"
)

errors = []

# 4a. Enforce 1-to-1 CuTeDSL top-level module layout in `cute-rs/src/`.
allowed_top_entries = {
    "arch.rs", "atom", "layout.rs", "lib.rs", "numeric.rs",
    "nvgpu", "pipeline", "tensor", "utils",
}
actual_top_entries = set(os.listdir("cuda-oxide/cute-rs-preview/cute-rs/src"))
unexpected_top = sorted(actual_top_entries - allowed_top_entries)
if unexpected_top:
    errors.append(
        f"cuda-oxide/cute-rs-preview/cute-rs/src: non-CuTeDSL top-level entries {unexpected_top} (allowed: {sorted(allowed_top_entries)})"
    )

# 4b. Enforce complete `cutlass-primitives` / `cutlass_primitives` ban inside `cute-rs`.
cargo_toml_path = "cuda-oxide/cute-rs-preview/cute-rs/Cargo.toml"
with open(cargo_toml_path, encoding="utf-8") as f:
    cargo_toml_text = f.read()
    if "cutlass-primitives" in cargo_toml_text or "ctm-rs" in cargo_toml_text:
        errors.append(f"{cargo_toml_path}: `cutlass-primitives` dependency is banned inside `cute-rs`")

for path in sorted(glob.glob("cuda-oxide/cute-rs-preview/cute-rs/src/**/*.rs", recursive=True)):
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if "cutlass_primitives" in line or "cutlass-primitives" in line or "ctm_rs" in line or "ctm-rs" in line:
                errors.append(f"{path}:{lineno}: `cutlass_primitives` is banned inside `cute-rs`: {line.strip()}")
            if "fn f32_to_e4m3" in line or "fn f32_to_e2m1" in line or "fn e4m3_to_f32" in line or "fn e2m1_to_f32" in line or "fn ue8m0_to_f32" in line:
                errors.append(f"{path}:{lineno}: software FP4/FP8 conversion helper is banned in `cute-rs` (lower through `__compiler` -> `dialect-cute`): {line.strip()}")
            if "/nvgpu/" in path and ("cuda_device" in line or "encode_no_swizzle_smem_desc" in line or "0x0000_0480" in line):
                errors.append(f"{path}:{lineno}: raw `cuda_device` or manual descriptor bit-packing is banned in `cute-rs/src/nvgpu/` (lower through `__compiler` -> `dialect-cute`): {line.strip()}")
            if path.endswith("tensor/mod.rs") and re.search(r"\bif\s+[NM]\s*==\s*(?:2|4|8|16)\b", line):
                errors.append(f"{path}:{lineno}: hardcoded `if N == 2/4/8/16` software unroll ladder is banned in `cute-rs/src/tensor/mod.rs` (lower through `__compiler` -> `dialect-cute` vector ops): {line.strip()}")

# 4c. Direct CuTeDSL Python AST symbol & workload-helper purity check across cute-rs/src/.
for root_dir in (
    "cuda-oxide/cute-rs-preview/cute-rs/src/tensor",
    "cuda-oxide/cute-rs-preview/cute-rs/src/nvgpu/tcgen05",
    "cuda-oxide/cute-rs-preview/cute-rs/src/pipeline",
    "cuda-oxide/cute-rs-preview/cute-rs/src/utils",
):
    for path in sorted(glob.glob(os.path.join(root_dir, "**", "*.rs"), recursive=True)):
        if path.endswith("__compiler.rs") or path.endswith("tests.rs"):
            continue
        in_test_mod = False
        in_compiler_mod = False
        with open(path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                if re.match(r"^\s*#\[cfg\(test\)\]", line) or re.match(r"^\s*mod\s+tests\b", line):
                    in_test_mod = True
                if re.match(r"^\s*pub\s+mod\s+__compiler\s*\{", line):
                    in_compiler_mod = True
                elif in_compiler_mod and line.startswith("}"):
                    in_compiler_mod = False
                if in_test_mod or in_compiler_mod:
                    continue
                if forbidden_cute_rs_re.search(line):
                    errors.append(f"{path}:{lineno}: workload-specific or non-CuTeDSL helper forbidden in cute-rs: {line.strip()}")
                if (
                    "/tensor/" in path
                    or "/nvgpu/tcgen05/" in path
                    or path.endswith("tmem_allocator.rs")
                    or path.endswith("pipeline/sm100.rs")
                ):
                    m = pub_fn_re.match(line)
                    if m and cutedsl_symbols:
                        fn_name = m.group(1)
                        if path.endswith("tensor/mod.rs") and fn_name in {
                            "make_tensor_read", "make_tensor_write",
                            "zipped_divide_read", "zipped_divide_write",
                            "slice_read", "slice_write",
                            "tensor_tile_is_full", "tensor_tile_base",
                            "tensor_load_tile", "tensor_store_tile",
                            "tensor_store_element_abs",
                        }:
                            continue
                        if fn_name not in cutedsl_symbols and fn_name not in rust_idiom_allowlist:
                            errors.append(
                                f"{path}:{lineno}: public function `{fn_name}` does not exist in CuTeDSL AST or cutedsl-comparison.md"
                            )

# 4d. Strict CuTe kernel purity check on nemotron3_moe_cute (main.rs and moe_192e.rs):
#     Inside device kernel modules (`mod kernels` / `pub mod kernels_192e`), forbid:
#     - any bit-shift operator (`<<` or `>>`),
#     - any scalar-unroll or bit-pack macro (`quant16_relu2`, `pack8`, `q_nib`, `st_relu2`, `st_dual16`, `st_triple16`),
#     - any raw `TmemFragment.values[...]` scalar indexing (use `to_tensor_ssa()` / `slice::<OFFSET, LEN>()`),
#     - any non-CuTeDSL `.to_u128()` or manual FP4/FP8 conversion helpers.
kernel_purity_re = re.compile(
    r"(?<!:)\s<<\s|\s>>\s"
    r"|\.values\["
    r"|\.to_u128\b"
    r"|macro_rules!\s*(?:quant16_relu2|pack8|q_nib|st_relu2|st_dual16|st_triple16)\b"
    r"|0xFFFF_FFFF|0xFFFF_0000"
    r"|\b(?:dev_e4m3_to_f32|dev_f32_to_e4m3|dev_e4m3_inv_e2m1|fast_pos_e2m1|fn\s+f32_to_e2m1|encode_e4m3_rne|exact_inv_scale_e2m1|encode_pos_e2m1|encode_signed_e2m1)\b"
)

# Self-test that `kernel_purity_re` catches all forbidden patterns:
for bad_sample in (
    "let x = lo | (hi << 32);",
    "let y = (s0 as u32) | ((s1 as u32) << 8);",
    "let z = tid >> 5;",
    "let v = frag.values[0];",
    "let u = tile.to_u128();",
    "macro_rules! quant16_relu2 {",
    "macro_rules! pack8 {",
):
    if not kernel_purity_re.search(bad_sample):
        sys.exit(f"error: kernel_purity_re self-test failed to catch: {bad_sample}")

for moe_path in (
    "cuda-oxide/cute-rs-preview/models/nemotron3_moe_cute/src/main.rs",
    "cuda-oxide/cute-rs-preview/models/nemotron3_moe_cute/src/moe_192e.rs",
):
    if os.path.isfile(moe_path):
        with open(moe_path, encoding="utf-8") as f:
            in_device_mod = False
            for lineno, line in enumerate(f, 1):
                if re.match(r"^\s*(?:pub\s+)?mod\s+kernels(?:_192e)?\s*\{", line):
                    in_device_mod = True
                elif in_device_mod and line.startswith("}"):
                    in_device_mod = False
                if in_device_mod and not line.lstrip().startswith("//"):
                    code = line.split("//")[0]
                    if kernel_purity_re.search(code):
                        errors.append(
                            f"{moe_path}:{lineno}: manual bit-shifting, scalar fragment unrolling, or non-CuTeDSL op in *_cute kernel (use TensorSSA / Layout algebra): {line.strip()}"
                        )

if errors:
    print("error: CuTeDSL source AST & cute-rs purity check failed:", file=sys.stderr)
    for err in errors:
        print(f"  {err}", file=sys.stderr)
    sys.exit(1)
PY
fi

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
