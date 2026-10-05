#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_file="${root}/src/main.rs"
llvm_ir="${1:-${root}/shared_debug.ll}"

# Keep standalone verification consistent with scripts/smoketest.sh: honor an
# explicit tool override and the conventional CUDA install roots even when
# ptxas is not on PATH.
ptxas_bin=""
for candidate in "${CUDA_OXIDE_PTXAS:-}" \
                 "$(command -v ptxas 2>/dev/null || true)" \
                 "${CUDA_HOME:+${CUDA_HOME}/bin/ptxas}" \
                 /usr/local/cuda/bin/ptxas \
                 /usr/local/cuda-*/bin/ptxas; do
    if [[ -n "${candidate}" && -x "${candidate}" ]]; then
        ptxas_bin="${candidate}"
        break
    fi
done

test -s "${llvm_ir}"

definition_line() {
    grep -nF "$1" "${source_file}" | cut -d: -f1
}

die_at_marker() {
    local name="$1" marker="$2" line
    line="$(definition_line "${marker}")"
    grep -F "!DIGlobalVariable(name: \"${name}\"" "${llvm_ir}" | grep -F "line: ${line},"
}

linkage_name() {
    sed -E 's/.*linkageName: "([^"]+)".*/\1/'
}

# A function-local static's expression has one verifier-accepted home per
# LLVM major: LLVM 23+ retains it on the owning DISubprogram's retainedNodes
# tuple, LLVM 21/22 list it in the compile unit's globals tuple. The exporter
# picks the form for the llc it resolved, so accept either here.
require_retained_class8() {
    local die="$1" die_id expression expression_id owner_id cu globals_id tuple
    die_id="$(printf '%s\n' "${die}" | sed -E 's/^!([0-9]+) = .*/\1/')"
    expression="$(grep -F "!DIGlobalVariableExpression(var: !${die_id}, expr: !DIExpression(DW_OP_constu, 8, DW_OP_swap, DW_OP_xderef))" "${llvm_ir}")"
    [[ "$(printf '%s\n' "${expression}" | wc -l)" -eq 1 ]]
    expression_id="$(printf '%s\n' "${expression}" | sed -E 's/^!([0-9]+) = .*/\1/')"
    owner_id="$(printf '%s\n' "${die}" | sed -E 's/.*scope: !([0-9]+).*/\1/')"
    if grep -Eq "^!${owner_id} = distinct !DISubprogram\\(.*retainedNodes: !\\{([^}]*[{ ])?!${expression_id}[,}]" "${llvm_ir}"; then
        record_placement retained
        return 0
    fi
    cu="$(grep -F '!DICompileUnit(' "${llvm_ir}")"
    globals_id="$(printf '%s\n' "${cu}" | sed -E 's/.*globals: !([0-9]+)\).*/\1/')"
    tuple="$(grep -E "^!${globals_id} = !\\{" "${llvm_ir}")"
    grep -Eq "[{ ]!${expression_id}[,}]" <<<"${tuple}"
    record_placement compile-unit
}

# Every function-local static must use the same placement.
placement=""
record_placement() {
    if [[ -n "${placement}" && "${placement}" != "$1" ]]; then
        echo "mixed function-local static placements: ${placement} and $1" >&2
        exit 1
    fi
    placement="$1"
}

require_physical() {
    local die="$1" type="$2" align="$3" linkage
    linkage="$(printf '%s\n' "${die}" | linkage_name)"
    [[ "$(grep -Ec "^@${linkage} = addrspace\\(3\\) global ${type} .*align ${align}, !dbg !" "${llvm_ir}")" -eq 1 ]]
}

tile_die="$(die_at_marker TILE DEBUG_SHARED_TILE)"
other_die="$(die_at_marker OTHER DEBUG_SHARED_OTHER)"
barrier_die="$(die_at_marker BAR DEBUG_SHARED_BARRIER)"
same_left_die="$(die_at_marker SAME DEBUG_SHARED_SAME_LEFT)"
same_right_die="$(die_at_marker SAME DEBUG_SHARED_SAME_RIGHT)"
other_scope_die="$(die_at_marker TILE DEBUG_SHARED_OTHER_SCOPE)"

for die in "${tile_die}" "${other_die}" "${barrier_die}" "${same_left_die}" "${same_right_die}" "${other_scope_die}"; do
    require_retained_class8 "${die}"
done

require_physical "${tile_die}" '\[32 x i32\]' 4
require_physical "${other_die}" '\[8 x i16\]' 2
require_physical "${barrier_die}" '\[1 x i64\]' 8
require_physical "${same_left_die}" '\[2 x i16\]' 2
require_physical "${same_right_die}" '\[2 x i16\]' 2
require_physical "${other_scope_die}" '\[4 x i64\]' 8

tile_type="$(printf '%s\n' "${tile_die}" | sed -E 's/.*type: !([0-9]+).*/\1/')"
grep -Eq "^!${tile_type} = !DICompositeType\\(tag: DW_TAG_array_type, baseType: ![0-9]+, size: 1024, elements: ![0-9]+\\)" "${llvm_ir}"
grep -Fq '!DISubrange(count: 32)' "${llvm_ir}"

barrier_type="$(printf '%s\n' "${barrier_die}" | sed -E 's/.*type: !([0-9]+).*/\1/')"
grep -Eq "^!${barrier_type} = !DICompositeType\\(tag: DW_TAG_structure_type, name: \"Barrier\", size: 64," "${llvm_ir}"

tile_scope="$(printf '%s\n' "${tile_die}" | sed -E 's/.*scope: !([0-9]+).*/\1/')"
other_scope="$(printf '%s\n' "${other_scope_die}" | sed -E 's/.*scope: !([0-9]+).*/\1/')"
same_left_scope="$(printf '%s\n' "${same_left_die}" | sed -E 's/.*scope: !([0-9]+).*/\1/')"
same_right_scope="$(printf '%s\n' "${same_right_die}" | sed -E 's/.*scope: !([0-9]+).*/\1/')"
grep -Eq "^!${tile_scope} = distinct !DISubprogram\\(name: \"shared_debug\", linkageName: \"[^\"]+\", scope: ![0-9]+," "${llvm_ir}"
grep -Eq "^!${other_scope} = distinct !DISubprogram\\(name: \"other_scope\", linkageName: \"[^\"]+\", scope: ![0-9]+," "${llvm_ir}"
grep -Eq "^!${same_left_scope} = distinct !DISubprogram\\(name: \"same_leaf\", linkageName: \"[^\"]+\", scope: ![0-9]+," "${llvm_ir}"
grep -Fq '!DINamespace(name: "shared_debug", scope: null)' "${llvm_ir}"
grep -Fq '!DINamespace(name: "kernels", scope: !' "${llvm_ir}"
[[ "$(grep -Ec 'distinct !DISubprogram\(name: "shared_debug"' "${llvm_ir}")" -eq 1 ]]
[[ "$(grep -Ec 'distinct !DISubprogram\(name: "other_scope"' "${llvm_ir}")" -eq 1 ]]
[[ "$(grep -Ec 'distinct !DISubprogram\(name: "same_leaf"' "${llvm_ir}")" -eq 1 ]]
[[ "${other_scope}" != "${tile_scope}" ]]
[[ "${same_left_scope}" == "${same_right_scope}" ]]
[[ "${same_left_scope}" != "${tile_scope}" ]]
grep -Eq "^define .*@shared_debug\\(.* !dbg !${tile_scope} \\{$" "${llvm_ir}"
grep -Eq "^define .*@other_scope\\(.* !dbg !${other_scope} \\{$" "${llvm_ir}"
grep -Eq "^define .*@same_leaf\\(.* !dbg !${same_left_scope} \\{$" "${llvm_ir}"

left_linkage="$(printf '%s\n' "${same_left_die}" | linkage_name)"
right_linkage="$(printf '%s\n' "${same_right_die}" | linkage_name)"
[[ "${left_linkage}" != "${right_linkage}" ]]
[[ "$(grep -Fc '!DIGlobalVariableExpression(var:' "${llvm_ir}")" -eq 6 ]]

tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT

# Verify the graph with LLVM tools whose major accepts the placement the
# exporter chose: LLVM 23+ for the retained-nodes form, LLVM 21/22 for the
# compile-unit form (the other major's verifier rejects the graph by design
# and, like llc, strips it with an exit status of 0, so stderr is checked
# too). Candidates are the `-NN`-suffixed tools on PATH and the Rust
# sysroot's llvm-tools. ptxas's cubin is authoritative for the NVPTX
# address-class translation; llvm-dwarfdump reads it, with readelf as the
# fallback when the toolset ships no dwarfdump.
toolsets=()
for version in 21 22 23 24 25; do
    if command -v "llc-${version}" >/dev/null 2>&1; then
        toolsets+=("|-${version}")
    fi
done
if command -v rustc >/dev/null 2>&1; then
    sysroot_bin="$(rustc --print sysroot)/lib/rustlib/$(rustc -vV | sed -n 's/^host: //p')/bin"
    if [[ -x "${sysroot_bin}/llc" ]]; then
        toolsets+=("${sysroot_bin}/|")
    fi
fi

reject_stripped_debug_info() {
    if grep -q 'invalid debug info' "$1"; then
        echo "$2 rejected the debug graph:" >&2
        cat "$1" >&2
        exit 1
    fi
}

validated=0
for entry in "${toolsets[@]}"; do
    prefix="${entry%%|*}"
    suffix="${entry##*|}"
    llvm_as="${prefix}llvm-as${suffix}"
    opt="${prefix}opt${suffix}"
    llc="${prefix}llc${suffix}"
    dwarfdump="${prefix}llvm-dwarfdump${suffix}"
    if ! command -v "${llvm_as}" >/dev/null 2>&1 \
        || ! command -v "${opt}" >/dev/null 2>&1 \
        || ! command -v "${llc}" >/dev/null 2>&1 \
        || [[ -z "${ptxas_bin}" ]]; then
        continue
    fi
    major="$("${llc}" --version | sed -nE 's/.*LLVM version ([0-9]+)\..*/\1/p' | head -n 1)"
    [[ -n "${major}" ]] || continue
    if [[ "${placement}" == retained && "${major}" -lt 23 ]] \
        || [[ "${placement}" == compile-unit && "${major}" -ge 23 ]]; then
        continue
    fi

    bitcode="${tmpdir}/shared-debug-${major}.bc"
    ptx="${tmpdir}/shared-debug-${major}.ptx"
    cubin="${tmpdir}/shared-debug-${major}.cubin"
    dwarf="${tmpdir}/shared-debug-${major}.dwarf"
    "${llvm_as}" -o "${bitcode}" "${llvm_ir}" 2>"${tmpdir}/as.err"
    reject_stripped_debug_info "${tmpdir}/as.err" "${llvm_as}"
    "${opt}" -passes=verify -disable-output "${bitcode}" 2>"${tmpdir}/opt.err"
    reject_stripped_debug_info "${tmpdir}/opt.err" "${opt}"
    "${llc}" -march=nvptx64 -mcpu=sm_90 -mattr=+ptx80 -O0 \
        -filetype=asm "${bitcode}" -o "${ptx}" 2>"${tmpdir}/llc.err"
    reject_stripped_debug_info "${tmpdir}/llc.err" "${llc}"
    grep -Eq '^\.target sm_90, debug$' "${ptx}"
    "${ptxas_bin}" -arch=sm_90 -g "${ptx}" -o "${cubin}"
    if command -v "${dwarfdump}" >/dev/null 2>&1; then
        "${dwarfdump}" --debug-info "${cubin}" >"${dwarf}"
        [[ "$(grep -c 'DW_AT_address_class.*0x08' "${dwarf}")" -eq 6 ]]
    else
        readelf --debug-dump=info "${cubin}" >"${dwarf}"
        [[ "$(grep -c 'DW_AT_address_class: 8$' "${dwarf}")" -eq 6 ]]
    fi
    validated=$((validated + 1))
done

if [[ "${validated}" -eq 0 ]]; then
    echo "shared-static AS3 debug-info: FAIL (no complete LLVM+ptxas toolset matching the ${placement} placement exercised cubin address class 8)" >&2
    exit 1
fi

echo "shared-static AS3 debug-info shape verified (placement: ${placement}; LLVM toolsets exercised: ${validated})"
