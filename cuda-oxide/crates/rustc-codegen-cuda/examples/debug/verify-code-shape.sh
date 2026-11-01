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
#
# Loop hints follow the same rule: nothing in `loop_hint_test` may map to a
# `#[unroll]` / `#[nounroll]` attribute line, and each loop body keeps its own.

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

node() {
    grep -E "^!${1} = " "${llvm_ir}" || true
}

# The LLVM IR body of kernel $1.
llvm_function_body() {
    awk -v name="$1" '
        index($0, "define ptx_kernel void @" name "(") == 1 { in_function = 1 }
        in_function { print }
        in_function && /^}/ { exit }
    ' "${llvm_ir}"
}

# The DISubprogram id of kernel $1.
kernel_subprogram() {
    grep -E "^![0-9]+ = distinct !DISubprogram\\(name: \"$1\"" "${llvm_ir}" |
        sed -nE 's/^!([0-9]+) = .*/\1/p' | head -n 1
}

# Does debug scope $1 belong to kernel subprogram $2 (not to an inlined callee)?
scope_is_in_kernel() {
    local scope="$1" kernel="$2" text
    for _ in 1 2 3 4 5 6 7 8 9 10 11 12; do
        [[ "${scope}" == "${kernel}" ]] && return 0
        text="$(node "${scope}")"
        grep -q 'DISubprogram(' <<<"${text}" && return 1
        scope="$(sed -nE 's/.*scope: !([0-9]+).*/\1/p' <<<"${text}")"
        [[ -n "${scope}" ]] || return 1
    done
    return 1
}

# Print the first of the `!dbg` ids in $1 whose location, or a location in its
# `inlinedAt` chain, lands on one of the lines in $2 within kernel subprogram $3.
first_location_on_lines() {
    local id location text line scope depth
    for id in $1; do
        location="${id}"
        for depth in 0 1 2 3 4 5 6 7; do
            text="$(node "${location}")"
            grep -q 'DILocation(' <<<"${text}" || break
            line="$(sed -nE 's/.*DILocation\(line: ([0-9]+),.*/\1/p' <<<"${text}")"
            scope="$(sed -nE 's/.*scope: !([0-9]+).*/\1/p' <<<"${text}")"
            if grep -qx "${line}" <<<"$2" && scope_is_in_kernel "${scope}" "$3"; then
                echo "!dbg !${id}, inlinedAt depth ${depth}, line ${line}"
                return 0
            fi
            location="$(sed -nE 's/.*inlinedAt: !([0-9]+)\).*/\1/p' <<<"${text}")"
            [[ -n "${location}" ]] || break
        done
    done
    return 0
}

# The PTX body of kernel entry $1, past any prototype declaration.
ptx_entry_body() {
    awk -v name="$1" '
        !emit && !candidate && $0 ~ ("[.]entry[[:space:]]+" name "[(]") { candidate = 1 }
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
    ' "${ptx}"
}

llvm_body="$(llvm_function_body clock_test)"
test -n "${llvm_body}"

kernel_subprogram_id="$(kernel_subprogram clock_test)"
test -n "${kernel_subprogram_id}"

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

prologue_hit="$(
    first_location_on_lines \
        "$(sed -nE 's/.*!dbg !([0-9]+).*/\1/p' <<<"${llvm_prologue}" | sort -u)" \
        "${kernel_attr_line}"$'\n'"${launch_bounds_attr_line}" \
        "${kernel_subprogram_id}"
)"
if [[ -n "${prologue_hit}" ]]; then
    echo "error: kernel prologue instruction (${prologue_hit}) maps to a macro attribute line" >&2
    exit 1
fi

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

ptx_body="$(ptx_entry_body clock_test)"
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

hint_attr_lines="$(grep -nF 'CUDA_OXIDE_DEBUG_LOOP_HINT_ATTRIBUTE_LINE' "${source_file}" | cut -d: -f1)"
hint_body_lines="$(grep -nF 'CUDA_OXIDE_DEBUG_LOOP_HINT_BODY_LINE' "${source_file}" | cut -d: -f1)"
test -n "${hint_attr_lines}"
test -n "${hint_body_lines}"

hint_llvm_body="$(llvm_function_body loop_hint_test)"
hint_subprogram_id="$(kernel_subprogram loop_hint_test)"
hint_ptx_body="$(ptx_entry_body loop_hint_test)"
test -n "${hint_llvm_body}"
test -n "${hint_subprogram_id}"
test -n "${hint_ptx_body}"

hint_hit="$(
    first_location_on_lines \
        "$(sed -nE 's/.*!dbg !([0-9]+).*/\1/p' <<<"${hint_llvm_body}" | sort -u)" \
        "${hint_attr_lines}" \
        "${hint_subprogram_id}"
)"
if [[ -n "${hint_hit}" ]]; then
    echo "error: loop_hint_test instruction (${hint_hit}) maps to a loop-hint attribute" >&2
    exit 1
fi

for line in ${hint_attr_lines}; do
    if grep -Eq "^[[:space:]]*[.]loc[[:space:]]+${main_file_index}[[:space:]]+${line}([[:space:]]|$)" <<<"${hint_ptx_body}"; then
        echo "error: loop_hint_test PTX maps an instruction to loop-hint attribute line ${line}" >&2
        exit 1
    fi
done
for line in ${hint_body_lines}; do
    if ! grep -Eq "^[[:space:]]*[.]loc[[:space:]]+${main_file_index}[[:space:]]+${line}([[:space:]]|$)" <<<"${hint_ptx_body}"; then
        echo "error: loop_hint_test PTX has no line-table entry for loop body line ${line}" >&2
        exit 1
    fi
done

echo "debug kernel-entry and loop-hint lineinfo shape: PASS"
