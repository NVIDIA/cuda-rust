#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build first with full device debug information:
#
#   cargo oxide build debug --device-debug
#
# This checks the kernel-entry line-table contract in both emitted LLVM IR
# and PTX. Source lines come from a same-line marker, never an assumed offset
# from `#[kernel]`; the fixture deliberately has documentation plus multiple
# attributes before the function.
#
# The contract is a negative one: nothing the compiler generates ahead of the
# user's first statement may be attributed to a macro attribute line. It must
# hold at any MIR optimization level. At mir-opt-level 0 the generated
# `make_kernel_scope` call survives and must carry an artificial (line 0)
# location; at the default level it is inlined away entirely (it returns a
# zero-sized value), so the check walks every instruction ahead of the index
# call, through its `inlinedAt` chain, and rejects any location that lands on
# an attribute line within the kernel's own scope.

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_file="${root}/src/main.rs"
llvm_ir="${root}/debug.ll"
ptx="${root}/debug.ptx"

test -s "${llvm_ir}"
test -s "${ptx}"

entry_line="$(grep -nF 'CUDA_OXIDE_DEBUG_KERNEL_ENTRY_LINE' "${source_file}" | cut -d: -f1)"
kernel_attr_line="$(grep -nF 'CUDA_OXIDE_DEBUG_KERNEL_ATTRIBUTE_LINE' "${source_file}" | cut -d: -f1)"
launch_bounds_attr_line="$(grep -nF 'CUDA_OXIDE_DEBUG_LAUNCH_BOUNDS_ATTRIBUTE_LINE' "${source_file}" | cut -d: -f1)"
test -n "${entry_line}"
test -n "${kernel_attr_line}"
test -n "${launch_bounds_attr_line}"

llvm_body="$(awk '
    /^define ptx_kernel void @clock_test\(/ { in_function = 1 }
    in_function { print }
    in_function && /^}/ { exit }
' "${llvm_ir}")"
test -n "${llvm_body}"

node() {
    grep -E "^!${1} = " "${llvm_ir}" || true
}

kernel_subprogram_id="$(
    grep -E '^![0-9]+ = distinct !DISubprogram\(name: "clock_test"' "${llvm_ir}" |
        sed -nE 's/^!([0-9]+) = .*/\1/p' | head -n 1
)"
test -n "${kernel_subprogram_id}"

# Does a debug scope belong to the kernel itself (not to an inlined callee)?
scope_is_in_kernel() {
    local scope="$1" text
    for _ in 1 2 3 4 5 6 7 8 9 10 11 12; do
        [[ "${scope}" == "${kernel_subprogram_id}" ]] && return 0
        text="$(node "${scope}")"
        grep -q 'DISubprogram(' <<<"${text}" && return 1
        scope="$(sed -nE 's/.*scope: !([0-9]+).*/\1/p' <<<"${text}")"
        [[ -n "${scope}" ]] || return 1
    done
    return 1
}

# Everything ahead of (and including) the rewritten index call.
llvm_prologue="$(awk '
    { print }
    /call i64 @.*index_1d/ { exit }
' <<<"${llvm_body}")"
index_debug_id="$(
    grep -m1 'call i64 @.*index_1d' <<<"${llvm_prologue}" |
        sed -nE 's/.*!dbg !([0-9]+).*/\1/p'
)"
test -n "${index_debug_id}"

for id in $(sed -nE 's/.*!dbg !([0-9]+).*/\1/p' <<<"${llvm_prologue}" | sort -u); do
    location="${id}"
    for depth in 0 1 2 3 4 5 6 7; do
        text="$(node "${location}")"
        grep -q 'DILocation(' <<<"${text}" || break
        line="$(sed -nE 's/.*DILocation\(line: ([0-9]+),.*/\1/p' <<<"${text}")"
        scope="$(sed -nE 's/.*scope: !([0-9]+).*/\1/p' <<<"${text}")"
        if [[ "${line}" == "${kernel_attr_line}" || "${line}" == "${launch_bounds_attr_line}" ]] \
            && scope_is_in_kernel "${scope}"; then
            echo "error: kernel prologue instruction (!dbg !${id}, inlinedAt depth ${depth}) maps to macro attribute line ${line}" >&2
            exit 1
        fi
        location="$(sed -nE 's/.*inlinedAt: !([0-9]+)\).*/\1/p' <<<"${text}")"
        [[ -n "${location}" ]] || break
    done
done

# A surviving (un-inlined) kernel-scope call must carry an artificial location.
scope_debug_id="$(
    { grep -m1 'call void @.*make_kernel_scope' <<<"${llvm_body}" || true; } |
        sed -nE 's/.*!dbg !([0-9]+).*/\1/p'
)"
if [[ -n "${scope_debug_id}" ]] \
    && ! grep -Eq "^!${scope_debug_id} = !DILocation\\(line: 0, column: 0," "${llvm_ir}"; then
    echo "error: generated kernel-scope call does not use an artificial LLVM location" >&2
    exit 1
fi
if ! grep -Eq "^!${index_debug_id} = !DILocation\\(line: ${entry_line}, column:" "${llvm_ir}"; then
    echo "error: rewritten index call lost source line ${entry_line} in LLVM IR" >&2
    exit 1
fi

ptx_body="$(awk '
    !emit && !candidate && /[.]entry[[:space:]]+clock_test\(/ { candidate = 1 }
    candidate && /^[[:space:]]*;[[:space:]]*$/ {
        candidate = 0
        next
    }
    candidate && /^[[:space:]]*\{[[:space:]]*$/ {
        emit = 1
        candidate = 0
    }
    emit { print }
    emit && /End function/ { exit }
' "${ptx}")"
test -n "${ptx_body}"

# PTX .loc directives name a file index; only main.rs lines can be attribute
# lines, so resolve its index instead of matching any file's line numbers.
main_file_index="$(
    sed -nE 's/^[[:space:]]*\.file[[:space:]]+([0-9]+)[[:space:]]+"[^"]*main\.rs".*/\1/p' "${ptx}" | head -n 1
)"
test -n "${main_file_index}"

ptx_prologue="$(awk '
    { print }
    /call[.]uni .*index_1d/ { exit }
' <<<"${ptx_body}")"
if grep -Eq "^[[:space:]]*[.]loc[[:space:]]+${main_file_index}[[:space:]]+(${kernel_attr_line}|${launch_bounds_attr_line})([[:space:]]|$)" <<<"${ptx_prologue}"; then
    echo "error: synthetic kernel prologue still maps to a macro attribute" >&2
    exit 1
fi

# A surviving kernel-scope call must sit under an artificial .loc; the
# rewritten index call must sit under the user's entry line either way.
if ! awk -v expected_index_line="${entry_line}" '
    $1 == ".loc" { current_line = $3 }
    /call[.]uni .*make_kernel_scope/ {
        saw_scope = 1
        scope_is_artificial = (current_line == 0)
    }
    /call[.]uni .*index_1d/ && !saw_index {
        saw_index = 1
        index_is_user_line = (current_line == expected_index_line)
    }
    END {
        exit !((!saw_scope || scope_is_artificial) && saw_index && index_is_user_line)
    }
' <<<"${ptx_body}"; then
    echo "error: PTX kernel prologue/index .loc state does not match the source contract" >&2
    exit 1
fi

echo "debug kernel-entry lineinfo shape: PASS"
