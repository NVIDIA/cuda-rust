#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build first with `CUDA_OXIDE_DEBUG=full cargo oxide build constant_memory_simple`.
# Set CUDA_OXIDE_VERIFY_CUDA_GDB=1 to also consume the result on a live GPU.

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_file="${root}/src/main.rs"
llvm_ir="${1:-${root}/constant_memory_simple.ll}"
retained_log_dir=""
if [[ -n "${CUDA_OXIDE_DEBUG_LOG_DIR:-}" ]]; then
    mkdir -p "${CUDA_OXIDE_DEBUG_LOG_DIR}"
    retained_log_dir="$(cd "${CUDA_OXIDE_DEBUG_LOG_DIR}" && pwd)"
fi

# Match the repository census's ptxas resolution. In particular, an explicit
# CUDA_OXIDE_PTXAS or a standard toolkit install must work even when the CUDA
# bin directory is intentionally absent from PATH.
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

fail() {
    echo "constant-memory debug-info: FAIL ($1)" >&2
    exit 1
}

[[ -s "${llvm_ir}" ]] || fail "missing LLVM IR at ${llvm_ir}"

unique_line() {
    local pattern="$1" line
    line="$(grep -F "${pattern}" "${llvm_ir}" || true)"
    [[ -n "${line}" ]] || fail "missing ${pattern}"
    [[ "$(printf '%s\n' "${line}" | wc -l)" -eq 1 ]] \
        || fail "${pattern} is not unique"
    printf '%s\n' "${line}"
}

metadata_id() {
    unique_line "$1" | sed -nE 's/^!([0-9]+) = .*/\1/p'
}

metadata_node() {
    unique_line "!$1 = "
}

scale_source_line="$(grep -nF 'static SCALE:' "${source_file}" | cut -d: -f1)"
[[ -n "${scale_source_line}" ]] || fail "SCALE source declaration not found"

crate_scope="$(metadata_id '!DINamespace(name: "constant_memory_simple", scope: null)')"
kernels_scope="$(metadata_id "!DINamespace(name: \"kernels\", scope: !${crate_scope})")"
scale_die="$(unique_line '!DIGlobalVariable(name: "SCALE"')"
grep -Fq "scope: !${kernels_scope}," <<<"${scale_die}" \
    || fail "SCALE is not scoped as constant_memory_simple::kernels::SCALE"
grep -Fq "line: ${scale_source_line}," <<<"${scale_die}" \
    || fail "SCALE declaration line is not preserved"
grep -Fq 'isLocal: false, isDefinition: true, align: 32)' <<<"${scale_die}" \
    || fail "SCALE visibility/alignment is wrong"

scale_die_id="$(sed -nE 's/^!([0-9]+) = .*/\1/p' <<<"${scale_die}")"
scale_type_id="$(sed -nE 's/.*type: !([0-9]+).*/\1/p' <<<"${scale_die}")"
scale_linkage="$(sed -nE 's/.*linkageName: "([^"]+)".*/\1/p' <<<"${scale_die}")"
[[ -n "${scale_die_id}" && -n "${scale_type_id}" && -n "${scale_linkage}" ]] \
    || fail "SCALE DIE is malformed"

scale_expression="$(unique_line "!DIGlobalVariableExpression(var: !${scale_die_id},")"
grep -Fq 'expr: !DIExpression())' <<<"${scale_expression}" \
    || fail "SCALE must use the empty AS4 DIExpression"
scale_expression_id="$(sed -nE 's/^!([0-9]+) = .*/\1/p' <<<"${scale_expression}")"

physical="$(unique_line "@${scale_linkage} = ")"
grep -Eq "^@${scale_linkage} = addrspace\\(4\\) global \\[4 x i8\\] .*align 4, !dbg !${scale_expression_id}$" \
    <<<"${physical}" || fail "SCALE physical AS4 storage is not attached to its expression"

# Follow the exact semantic type graph. The physical [4 x i8] backing must be
# described as ConstantMemory -> field 0 -> UnsafeCell -> value: f32. Generic
# ADT arguments are an existing presentation limitation of the shared builder.
wrapper="$(metadata_node "${scale_type_id}")"
grep -Fq 'DICompositeType(tag: DW_TAG_structure_type, name: "ConstantMemory", size: 32,' \
    <<<"${wrapper}" || fail "SCALE type is not ConstantMemory"
wrapper_elements_id="$(sed -nE 's/.*elements: !([0-9]+)\).*/\1/p' <<<"${wrapper}")"
wrapper_elements="$(metadata_node "${wrapper_elements_id}")"
wrapper_member_id="$(sed -nE 's/^![0-9]+ = !\{!([0-9]+)\}$/\1/p' <<<"${wrapper_elements}")"
wrapper_member="$(metadata_node "${wrapper_member_id}")"
grep -Fq 'DW_TAG_member, name: "0"' <<<"${wrapper_member}" \
    && grep -Fq 'size: 32, offset: 0)' <<<"${wrapper_member}" \
    || fail "ConstantMemory wrapper field size/offset is wrong"
unsafe_cell_id="$(sed -nE 's/.*baseType: !([0-9]+).*/\1/p' <<<"${wrapper_member}")"
unsafe_cell="$(metadata_node "${unsafe_cell_id}")"
grep -Fq 'DICompositeType(tag: DW_TAG_structure_type, name: "UnsafeCell", size: 32,' \
    <<<"${unsafe_cell}" || fail "ConstantMemory payload is not UnsafeCell"
unsafe_elements_id="$(sed -nE 's/.*elements: !([0-9]+)\).*/\1/p' <<<"${unsafe_cell}")"
unsafe_elements="$(metadata_node "${unsafe_elements_id}")"
value_member_id="$(sed -nE 's/^![0-9]+ = !\{!([0-9]+)\}$/\1/p' <<<"${unsafe_elements}")"
value_member="$(metadata_node "${value_member_id}")"
grep -Fq 'DW_TAG_member, name: "value"' <<<"${value_member}" \
    && grep -Fq 'size: 32, offset: 0)' <<<"${value_member}" \
    || fail "UnsafeCell value field size/offset is wrong"
f32_id="$(sed -nE 's/.*baseType: !([0-9]+).*/\1/p' <<<"${value_member}")"
grep -Fq '!DIBasicType(name: "f32", size: 32, encoding: DW_ATE_float)' \
    <<<"$(metadata_node "${f32_id}")" || fail "SCALE payload is not f32"

compile_unit="$(unique_line '!DICompileUnit(')"
globals_id="$(sed -nE 's/.*globals: !([0-9]+)\).*/\1/p' <<<"${compile_unit}")"
[[ -n "${globals_id}" ]] || fail "compile unit does not retain globals"
grep -Eq "[{ ]!${scale_expression_id}[,}]" <<<"$(metadata_node "${globals_id}")" \
    || fail "compile unit does not retain SCALE"

# UNUSED is deliberately unreachable. The positive AS4 path must not invent a
# DIE for a static that the importer never materialized.
! grep -Fq '!DIGlobalVariable(name: "UNUSED"' "${llvm_ir}" \
    || fail "unmaterialized UNUSED unexpectedly acquired a DIE"
! grep -Fq 'DW_OP_constu, 4' "${llvm_ir}" \
    || fail "AS4 must not use a hand-authored address-class expression"

tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT

toolsets=()
if command -v llvm-as >/dev/null 2>&1 \
    && command -v opt >/dev/null 2>&1 \
    && command -v llc >/dev/null 2>&1; then
    toolsets+=("|")
fi
if command -v rustc >/dev/null 2>&1; then
    sysroot_bin="$(rustc --print sysroot)/lib/rustlib/$(rustc -vV | sed -n 's/^host: //p')/bin"
    if [[ -x "${sysroot_bin}/llvm-as" && -x "${sysroot_bin}/opt" && -x "${sysroot_bin}/llc" ]]; then
        toolsets+=("${sysroot_bin}/|")
    fi
fi
for version in 21 22 23 24 25; do
    if command -v "llvm-as-${version}" >/dev/null 2>&1 \
        && command -v "opt-${version}" >/dev/null 2>&1 \
        && command -v "llc-${version}" >/dev/null 2>&1; then
        toolsets+=("|-${version}")
    fi
done

validated=0
declare -A seen_llvm_majors=()
if [[ -n "${ptxas_bin}" ]]; then
    for entry in "${toolsets[@]}"; do
        prefix="${entry%%|*}"
        suffix="${entry##*|}"
        llvm_as="${prefix}llvm-as${suffix}"
        opt="${prefix}opt${suffix}"
        llc="${prefix}llc${suffix}"
        major="$("${llc}" --version | sed -nE 's/.*LLVM version ([0-9]+)\..*/\1/p' | head -n 1)"
        [[ -n "${major}" ]] || continue
        # CUDA Oxide's supported/export-tested debug metadata starts at LLVM
        # 21. An older distro-default tool must not turn a valid LLVM 23
        # artifact into a false failure, and duplicate aliases count once.
        [[ "${major}" -ge 21 ]] || continue
        [[ -z "${seen_llvm_majors[${major}]+x}" ]] || continue
        seen_llvm_majors["${major}"]=1

        bitcode="${tmpdir}/constant-memory-${major}.bc"
        ptx="${tmpdir}/constant-memory-${major}.ptx"
        cubin="${tmpdir}/constant-memory-${major}.cubin"
        dwarf="${tmpdir}/constant-memory-${major}.dwarf"
        "${llvm_as}" -o "${bitcode}" "${llvm_ir}" 2>"${tmpdir}/as-${major}.err"
        ! grep -q 'invalid debug info' "${tmpdir}/as-${major}.err" \
            || fail "${llvm_as} rejected the AS4 debug graph"
        "${opt}" -passes=verify -disable-output "${bitcode}" 2>"${tmpdir}/opt-${major}.err"
        ! grep -q 'invalid debug info' "${tmpdir}/opt-${major}.err" \
            || fail "${opt} rejected the AS4 debug graph"
        "${llc}" -march=nvptx64 -mcpu=sm_90 -mattr=+ptx80 -O0 \
            -filetype=asm "${bitcode}" -o "${ptx}" 2>"${tmpdir}/llc-${major}.err"
        ! grep -q 'invalid debug info' "${tmpdir}/llc-${major}.err" \
            || fail "${llc} stripped the AS4 debug graph"
        grep -Eq '^\.target sm_90, debug$' "${ptx}" \
            || fail "${llc} did not retain device debug mode"
        grep -Eq "^[.]visible [.](const).* ${scale_linkage}(\\[4\\])?;" "${ptx}" \
            || fail "PTX does not contain SCALE constant storage"
        "${ptxas_bin}" -arch=sm_90 -g "${ptx}" -o "${cubin}"

        dwarf_tool=""
        if command -v "${prefix}llvm-dwarfdump${suffix}" >/dev/null 2>&1; then
            dwarf_tool="${prefix}llvm-dwarfdump${suffix}"
            "${dwarf_tool}" --debug-info "${cubin}" >"${dwarf}"
            [[ "$(grep -c 'DW_AT_address_class.*(0x04)' "${dwarf}")" -eq 1 ]] \
                || fail "cubin does not contain exactly one AS4 variable"
        elif command -v llvm-dwarfdump >/dev/null 2>&1; then
            dwarf_tool="llvm-dwarfdump"
            "${dwarf_tool}" --debug-info "${cubin}" >"${dwarf}"
            [[ "$(grep -c 'DW_AT_address_class.*(0x04)' "${dwarf}")" -eq 1 ]] \
                || fail "cubin does not contain exactly one AS4 variable"
        else
            fail "llvm-dwarfdump is required to inspect and associate SCALE with its cubin address class"
        fi
        scale_dwarf="${tmpdir}/constant-memory-${major}-scale.dwarf"
        "${dwarf_tool}" --debug-info --name SCALE "${cubin}" >"${scale_dwarf}"
        [[ "$(grep -c 'DW_TAG_variable' "${scale_dwarf}")" -eq 1 ]] \
            || fail "cubin does not contain exactly one SCALE variable DIE"
        grep -Fq 'DW_AT_name' "${scale_dwarf}" \
            && grep -Fq 'SCALE' "${scale_dwarf}" \
            || fail "cubin DWARF lost the SCALE name"
        grep -Fq 'DW_AT_address_class' "${scale_dwarf}" \
            && grep -Fq '(0x04)' "${scale_dwarf}" \
            || fail "SCALE variable DIE is not tagged with AS4 address class"
        grep -Fq 'DW_AT_type' "${scale_dwarf}" \
            && grep -Fq 'ConstantMemory' "${scale_dwarf}" \
            || fail "SCALE variable DIE lost the semantic ConstantMemory type"
        validated=$((validated + 1))
    done
fi

if [[ "${validated}" -eq 0 ]]; then
    fail "no complete LLVM+ptxas toolset exercised the cubin DWARF"
fi

if [[ "${CUDA_OXIDE_VERIFY_CUDA_GDB:-0}" == 1 ]]; then
    cuda_gdb="${CUDA_OXIDE_CUDA_GDB:-$(command -v cuda-gdb || true)}"
    [[ -x "${cuda_gdb}" ]] || fail "CUDA_OXIDE_VERIFY_CUDA_GDB=1 but cuda-gdb is unavailable"
    command -v nvidia-smi >/dev/null 2>&1 \
        || fail "CUDA_OXIDE_VERIFY_CUDA_GDB=1 but nvidia-smi is unavailable"
    nvidia-smi -L >/dev/null 2>&1 \
        || fail "CUDA_OXIDE_VERIFY_CUDA_GDB=1 but no GPU is usable"
    binary="${root}/target/release/constant_memory_simple"
    [[ -x "${binary}" ]] || fail "missing host binary at ${binary}"

    live_dir="${tmpdir}/live"
    mkdir -p "${live_dir}/src"
    cp "${binary}" "${live_dir}/constant_memory_simple"
    cp "${root}/constant_memory_simple.ptx" "${live_dir}/constant_memory_simple.ptx"
    cp "${source_file}" "${live_dir}/src/main.rs"
    if [[ -n "${retained_log_dir}" ]]; then
        runtime_log="${retained_log_dir}/constant_memory_simple-runtime.log"
        gdb_log="${retained_log_dir}/constant_memory_simple-cuda-gdb.log"
        : >"${runtime_log}"
        : >"${gdb_log}"
        echo "constant-memory debug-info: retaining complete logs in ${retained_log_dir}"
    else
        runtime_log="${tmpdir}/runtime.log"
        gdb_log="${tmpdir}/cuda-gdb.log"
    fi
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
    (cd "${live_dir}" && ./constant_memory_simple) >"${runtime_log}" 2>&1 \
        || fail "constant-memory runtime control failed"
    grep -Fq '[0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0]' "${runtime_log}" \
        || fail "constant-memory runtime values changed"
    grep -Fq 'SUCCESS: constant-memory scale applied correctly (8 elements)' "${runtime_log}" \
        || fail "constant-memory runtime success marker is missing"
    (
        cd "${live_dir}"
        timeout 300 "${cuda_gdb}" --batch \
            -ex 'set pagination off' \
            -ex 'set confirm off' \
            -ex 'set cuda break_on_launch application' \
            -ex 'run' \
            -ex 'set language rust' \
            -ex 'show language' \
            -ex 'print constant_memory_simple::kernels::SCALE' \
            -ex 'ptype constant_memory_simple::kernels::SCALE' \
            -ex 'backtrace' \
            -ex 'kill' \
            ./constant_memory_simple
    ) >"${gdb_log}" 2>&1 || fail "cuda-gdb session failed"
    if ! grep -qiE 'CUDA thread hit|Breakpoint .*multiply' "${gdb_log}"; then
        tail -80 "${gdb_log}" >&2
        fail "cuda-gdb did not stop in the kernel"
    fi
    if ! grep -Eq '= ConstantMemory \{0: UnsafeCell \{value: 3([.]0*)?\}\}' "${gdb_log}"; then
        tail -80 "${gdb_log}" >&2
        fail "Rust-qualified SCALE did not resolve to the initialized value"
    fi
    if ! grep -Fq 'type = struct ConstantMemory' "${gdb_log}"; then
        tail -80 "${gdb_log}" >&2
        fail "cuda-gdb lost SCALE's semantic type"
    fi
    grep -Fq 'The current source language is "rust".' "${gdb_log}" \
        || fail "constant-memory lookup did not run in Rust language mode"
    grep -Eq '^#0 .*multiply.*<<<.*>>>' "${gdb_log}" \
        || fail "constant-memory lookup is not in the kernel frame"
    ! grep -qiE 'INVALID_PTX|JIT compilation failed|No device code' "${gdb_log}" \
        || fail "device code failed to load under cuda-gdb"

    echo "----- constant-memory cuda-gdb output (tail) -----"
    tail -30 "${gdb_log}"
    echo "--------------------------------------------------"
fi

echo "constant-memory AS4 debug-info: PASS (LLVM toolsets exercised: ${validated})"
