#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_file="${root}/src/main.rs"
llvm_ir="${root}/device_global.ll"

test -s "${llvm_ir}"

definition_line() {
    local marker="$1"
    grep -nF "${marker}" "${source_file}" | cut -d: -f1
}

metadata_id() {
    local pattern="$1"
    local line
    line="$(grep -F "${pattern}" "${llvm_ir}")"
    [[ -n "${line}" ]]
    [[ "$(printf '%s\n' "${line}" | wc -l)" -eq 1 ]]
    printf '%s\n' "${line}" | sed -E 's/^!([0-9]+) = .*/\1/'
}

unique_global_die() {
    local source_name="$1"
    local line
    line="$(grep -F "!DIGlobalVariable(name: \"${source_name}\"" "${llvm_ir}")"
    [[ -n "${line}" ]]
    [[ "$(printf '%s\n' "${line}" | wc -l)" -eq 1 ]]
    printf '%s\n' "${line}"
}

named_global_die_at_marker() {
    local source_name="$1"
    local marker="$2"
    local source_line line
    source_line="$(definition_line "${marker}")"
    line="$(grep -F "!DIGlobalVariable(name: \"${source_name}\"" "${llvm_ir}" | grep -F "line: ${source_line},")"
    [[ -n "${line}" ]]
    [[ "$(printf '%s\n' "${line}" | wc -l)" -eq 1 ]]
    printf '%s\n' "${line}"
}

linkage_name() {
    sed -E 's/.*linkageName: "([^"]+)".*/\1/'
}

require_cu_expression() {
    local die="$1"
    local die_id expression expression_id cu globals_id tuple
    die_id="$(printf '%s\n' "${die}" | sed -E 's/^!([0-9]+) = .*/\1/')"
    expression="$(grep -F "!DIGlobalVariableExpression(var: !${die_id}," "${llvm_ir}")"
    [[ -n "${expression}" ]]
    [[ "$(printf '%s\n' "${expression}" | wc -l)" -eq 1 ]]
    expression_id="$(printf '%s\n' "${expression}" | sed -E 's/^!([0-9]+) = .*/\1/')"
    cu="$(grep -F '!DICompileUnit(' "${llvm_ir}")"
    [[ -n "${cu}" ]]
    [[ "$(printf '%s\n' "${cu}" | wc -l)" -eq 1 ]]
    globals_id="$(printf '%s\n' "${cu}" | sed -E 's/.*globals: !([0-9]+)\).*/\1/')"
    tuple="$(grep -E "^!${globals_id} = !\\{" "${llvm_ir}")"
    [[ -n "${tuple}" ]]
    grep -Fq "!${expression_id}" <<<"${tuple}"
}

crate_scope="$(metadata_id '!DINamespace(name: "device_global", scope: null)')"
left_scope="$(metadata_id "!DINamespace(name: \"debug_left\", scope: !${crate_scope})")"
right_scope="$(metadata_id "!DINamespace(name: \"debug_right\", scope: !${crate_scope})")"

mapfile -t same_leaf_dies < <(grep -F '!DIGlobalVariable(name: "SAME_LEAF"' "${llvm_ir}")
[[ "${#same_leaf_dies[@]}" -eq 2 ]]
left_die="$(printf '%s\n' "${same_leaf_dies[@]}" | grep -F "scope: !${left_scope}")"
right_die="$(printf '%s\n' "${same_leaf_dies[@]}" | grep -F "scope: !${right_scope}")"
grep -Fq "line: $(definition_line CUDA_OXIDE_DEBUG_GLOBAL_LEFT)," <<<"${left_die}"
grep -Fq "line: $(definition_line CUDA_OXIDE_DEBUG_GLOBAL_RIGHT)," <<<"${right_die}"
grep -Fq 'isLocal: true' <<<"${left_die}"
grep -Fq 'isLocal: true' <<<"${right_die}"

left_linkage="$(printf '%s\n' "${left_die}" | linkage_name)"
right_linkage="$(printf '%s\n' "${right_die}" | linkage_name)"
[[ "${left_linkage}" != "${right_linkage}" ]]
[[ "$(grep -Ec "^@${left_linkage} = " "${llvm_ir}")" -eq 1 ]]
[[ "$(grep -Ec "^@${right_linkage} = " "${llvm_ir}")" -eq 1 ]]
require_cu_expression "${left_die}"
require_cu_expression "${right_die}"

# Two statics in opposite blocks of one function deliberately have identical
# `StaticDef::name()` display paths. Their DefPath-disambiguated identity keys
# must produce separate storage and DIEs, and each reference initializer must
# retain provenance to the matching value rather than the other block's value.
block_local_scope="$(metadata_id "!DINamespace(name: \"block_local_static_values\", scope: !${crate_scope})")"

mapfile -t block_value_dies < <(grep -F '!DIGlobalVariable(name: "VALUE"' "${llvm_ir}")
mapfile -t block_reference_dies < <(grep -F '!DIGlobalVariable(name: "VALUE_REF"' "${llvm_ir}")
[[ "${#block_value_dies[@]}" -eq 2 ]]
[[ "${#block_reference_dies[@]}" -eq 2 ]]

left_value_die="$(named_global_die_at_marker VALUE CUDA_OXIDE_DEBUG_BLOCK_LOCAL_LEFT_VALUE)"
right_value_die="$(named_global_die_at_marker VALUE CUDA_OXIDE_DEBUG_BLOCK_LOCAL_RIGHT_VALUE)"
left_reference_die="$(named_global_die_at_marker VALUE_REF CUDA_OXIDE_DEBUG_BLOCK_LOCAL_LEFT_REFERENCE)"
right_reference_die="$(named_global_die_at_marker VALUE_REF CUDA_OXIDE_DEBUG_BLOCK_LOCAL_RIGHT_REFERENCE)"

for die in "${left_value_die}" "${right_value_die}" "${left_reference_die}" "${right_reference_die}"; do
    grep -Fq "scope: !${block_local_scope}" <<<"${die}"
    grep -Fq 'isLocal: true' <<<"${die}"
    require_cu_expression "${die}"
done

left_value_linkage="$(printf '%s\n' "${left_value_die}" | linkage_name)"
right_value_linkage="$(printf '%s\n' "${right_value_die}" | linkage_name)"
left_reference_linkage="$(printf '%s\n' "${left_reference_die}" | linkage_name)"
right_reference_linkage="$(printf '%s\n' "${right_reference_die}" | linkage_name)"

[[ "${left_value_linkage}" != "${right_value_linkage}" ]]
[[ "${left_reference_linkage}" != "${right_reference_linkage}" ]]
for linkage in "${left_value_linkage}" "${right_value_linkage}" "${left_reference_linkage}" "${right_reference_linkage}"; do
    [[ "$(grep -Ec "^@${linkage} = " "${llvm_ir}")" -eq 1 ]]
done

left_value_definition="$(grep -E "^@${left_value_linkage} = " "${llvm_ir}")"
right_value_definition="$(grep -E "^@${right_value_linkage} = " "${llvm_ir}")"
grep -Fq '[8 x i8] c"\0B\00\00\00\00\00\00\00"' <<<"${left_value_definition}"
grep -Fq '[8 x i8] c"\1D\00\00\00\00\00\00\00"' <<<"${right_value_definition}"

[[ "$(grep -Ec "^@${left_reference_linkage} = .*@${left_value_linkage}.*!dbg !" "${llvm_ir}")" -eq 1 ]]
[[ "$(grep -Ec "^@${right_reference_linkage} = .*@${right_value_linkage}.*!dbg !" "${llvm_ir}")" -eq 1 ]]
! grep -Eq "^@${left_reference_linkage} = .*@${right_value_linkage}" "${llvm_ir}"
! grep -Eq "^@${right_reference_linkage} = .*@${left_value_linkage}" "${llvm_ir}"

left_value_type="$(printf '%s\n' "${left_value_die}" | sed -E 's/.*type: !([0-9]+).*/\1/')"
right_value_type="$(printf '%s\n' "${right_value_die}" | sed -E 's/.*type: !([0-9]+).*/\1/')"
[[ "${left_value_type}" == "${right_value_type}" ]]
grep -Eq "^!${left_value_type} = !DIBasicType\(name: \"u64\", size: 64, encoding: DW_ATE_unsigned\)" "${llvm_ir}"

left_reference_type="$(printf '%s\n' "${left_reference_die}" | sed -E 's/.*type: !([0-9]+).*/\1/')"
right_reference_type="$(printf '%s\n' "${right_reference_die}" | sed -E 's/.*type: !([0-9]+).*/\1/')"
[[ "${left_reference_type}" == "${right_reference_type}" ]]
grep -Eq "^!${left_reference_type} = !DIDerivedType\(tag: DW_TAG_pointer_type, name: \"&u64\", .*size: 64\)" "${llvm_ir}"

reachable_die="$(unique_global_die DEBUG_REACHABLE)"
private_die="$(unique_global_die DEBUG_PRIVATE)"
grep -Fq "line: $(definition_line CUDA_OXIDE_DEBUG_GLOBAL_REACHABLE)," <<<"${reachable_die}"
grep -Fq "line: $(definition_line CUDA_OXIDE_DEBUG_GLOBAL_PRIVATE)," <<<"${private_die}"
grep -Fq 'isLocal: false' <<<"${reachable_die}"
grep -Fq 'isLocal: true' <<<"${private_die}"
require_cu_expression "${reachable_die}"
require_cu_expression "${private_die}"

target_die="$(unique_global_die RELOCATION_TARGET_A)"
reference_die="$(unique_global_die RELOCATION_REFERENCE)"
target_line="$(grep -nF 'static RELOCATION_TARGET_A:' "${source_file}" | cut -d: -f1)"
reference_line="$(grep -nF 'static RELOCATION_REFERENCE:' "${source_file}" | cut -d: -f1)"
grep -Fq "line: ${target_line}," <<<"${target_die}"
grep -Fq "line: ${reference_line}," <<<"${reference_die}"
target_linkage="$(printf '%s\n' "${target_die}" | linkage_name)"
reference_linkage="$(printf '%s\n' "${reference_die}" | linkage_name)"
[[ "$(grep -Ec "^@${target_linkage} = .*\\[4 x i8\\].*!dbg !" "${llvm_ir}")" -eq 1 ]]
[[ "$(grep -Ec "^@${reference_linkage} = .*@${target_linkage}.*!dbg !" "${llvm_ir}")" -eq 1 ]]
reference_type="$(printf '%s\n' "${reference_die}" | sed -E 's/.*type: !([0-9]+).*/\1/')"
grep -Eq "^!${reference_type} = !DIDerivedType\\(tag: DW_TAG_pointer_type, name: \"&u32\", .*size: 64\\)" "${llvm_ir}"
require_cu_expression "${target_die}"
require_cu_expression "${reference_die}"

# The shared semantic type builder cannot yet describe unions or fat
# references exactly. Those individual globals fail closed; their physical
# storage remains present and Full debug still succeeds for every supported
# AS1 neighbor.
! grep -Fq '!DIGlobalVariable(name: "UNION_RELOCATION"' "${llvm_ir}"
! grep -Fq '!DIGlobalVariable(name: "SLICE_RELOCATION_VIEW"' "${llvm_ir}"

if command -v llvm-as >/dev/null 2>&1; then
    bitcode="$(mktemp --suffix=.bc)"
    trap 'rm -f "${bitcode}"' EXIT
    llvm-as -o "${bitcode}" "${llvm_ir}"
    if command -v opt >/dev/null 2>&1; then
        opt -passes=verify -disable-output "${bitcode}"
    fi
fi

echo "device-global debug-info shape verified"
