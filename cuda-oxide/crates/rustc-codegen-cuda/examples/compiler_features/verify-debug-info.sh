#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build first with `CUDA_OXIDE_DEBUG=full cargo oxide build compiler_features`.
# This verifier checks the bounded T08/T14 contract in the assembled cubin:
# DisjointSlice::get_mut is a physical function with one contiguous range, and
# test_option is a physical kernel subprogram.  It deliberately does not claim
# that arbitrary inlined NVPTX functions have correct discontiguous ranges.

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ptx="${1:-${root}/compiler_features.ptx}"

fail() {
    echo "compiler-features debug-info: FAIL ($1)" >&2
    exit 1
}

[[ -s "${ptx}" ]] || fail "missing PTX at ${ptx}"
grep -Eq '^\.target[[:space:]]+sm_[0-9]+[af]?,[[:space:]]*debug([[:space:]]*,.*)?$' "${ptx}" \
    || fail "PTX is not a concrete full-debug target"
grep -Eq '^[[:space:]]*\.section[[:space:]]+\.debug_info([[:space:]]|$)' "${ptx}" \
    || fail "PTX has no .debug_info section"

arch="$(sed -nE 's/^\.target[[:space:]]+(sm_[0-9]+[af]?),[[:space:]]*debug.*/\1/p' "${ptx}" | head -1)"
[[ -n "${arch}" ]] || fail "could not resolve PTX target architecture"

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
[[ -n "${ptxas_bin}" ]] || fail "ptxas not found (set CUDA_OXIDE_PTXAS)"

dwarfdump_bin=""
if [[ -n "${CUDA_OXIDE_LLVM_DWARFDUMP:-}" ]]; then
    [[ -x "${CUDA_OXIDE_LLVM_DWARFDUMP}" ]] \
        || fail "CUDA_OXIDE_LLVM_DWARFDUMP is not executable"
    dwarfdump_bin="${CUDA_OXIDE_LLVM_DWARFDUMP}"
else
    for candidate in llvm-dwarfdump-25 llvm-dwarfdump-24 llvm-dwarfdump-23 \
                     llvm-dwarfdump-22 llvm-dwarfdump-21 llvm-dwarfdump-20 \
                     llvm-dwarfdump; do
        if command -v "${candidate}" >/dev/null 2>&1; then
            dwarfdump_bin="$(command -v "${candidate}")"
            break
        fi
    done
fi
[[ -n "${dwarfdump_bin}" ]] \
    || fail "llvm-dwarfdump 20 or newer not found (set CUDA_OXIDE_LLVM_DWARFDUMP)"
dwarfdump_major="$("${dwarfdump_bin}" --version | sed -nE 's/.*LLVM version ([0-9]+).*/\1/p' | head -1)"
[[ -n "${dwarfdump_major}" && "${dwarfdump_major}" -ge 20 ]] \
    || fail "llvm-dwarfdump 20 or newer is required"

# Validate each returned DIE independently.  A query can return multiple
# get_mut monomorphizations; every one must be a concrete, contiguous
# subprogram.  low_pc == 0 is valid in CUDA cubins because each function can
# have its own text section, but high_pc must be strictly greater.
validate_concrete_subprograms() {
    local label="$1" minimum="$2" maximum="$3"
    awk -v label="${label}" -v minimum="${minimum}" -v maximum="${maximum}" '
        function normalized_hex(text) {
            sub(/^0x/, "", text)
            text = tolower(text)
            sub(/^0+/, "", text)
            return text == "" ? "0" : text
        }
        function hex_is_greater(left, right, normalized_left, normalized_right) {
            normalized_left = normalized_hex(left)
            normalized_right = normalized_hex(right)
            if (length(normalized_left) != length(normalized_right))
                return length(normalized_left) > length(normalized_right)
            # Prefix both operands so awk cannot coerce an all-decimal-looking
            # 64-bit address to an imprecise floating-point number.
            return ("x" normalized_left) > ("x" normalized_right)
        }
        function report(message) {
            print label ": " message > "/dev/stderr"
            errors++
        }
        function finish_die() {
            if (!in_die) return
            count++
            if (tag != "DW_TAG_subprogram")
                report("matching DIE " count " is " tag ", not DW_TAG_subprogram")
            if (forbidden)
                report("matching DIE " count " uses inline/range-list attributes")
            if (low_count != 1)
                report("matching DIE " count " has " low_count " DW_AT_low_pc attributes")
            if (high_count != 1)
                report("matching DIE " count " has " high_count " DW_AT_high_pc attributes")
            if (low_count == 1 && high_count == 1) {
                if (low == "" || high == "") {
                    report("matching DIE " count " has an unparseable PC attribute")
                    return
                }
                if (!hex_is_greater(high, low))
                    report("matching DIE " count " has a non-positive PC range " low ".." high)
            }
        }
        /^[[:space:]]*0x[[:xdigit:]]+:[[:space:]]+DW_TAG_/ {
            finish_die()
            in_die = 1
            tag = $2
            low = high = ""
            low_count = high_count = forbidden = 0
            next
        }
        in_die && /DW_AT_low_pc/ {
            low_count++
            if (match($0, /0x[[:xdigit:]]+/)) low = substr($0, RSTART, RLENGTH)
            next
        }
        in_die && /DW_AT_high_pc/ {
            high_count++
            if (match($0, /0x[[:xdigit:]]+/)) high = substr($0, RSTART, RLENGTH)
            next
        }
        in_die && /(DW_AT_inline|DW_AT_ranges)/ { forbidden = 1 }
        END {
            finish_die()
            if (count < minimum)
                report("query returned " count " matching DIEs; expected at least " minimum)
            if (maximum > 0 && count > maximum)
                report("query returned " count " matching DIEs; expected at most " maximum)
            exit(errors != 0)
        }
    '
}

# Keep the parser itself honest.  These controls catch the failure modes that
# motivated this gate: empty/abstract results, inline/range-list DIEs, and
# missing, empty, or reversed concrete ranges.
valid_sample=$'0x00000001: DW_TAG_subprogram\n              DW_AT_low_pc (0x0000000000000000)\n              DW_AT_high_pc (0x0000000000000010)'
validate_concrete_subprograms parser-control 1 1 <<<"${valid_sample}" >/dev/null 2>&1 \
    || fail "internal parser rejected its valid control"
large_valid_sample=$'0x00000001: DW_TAG_subprogram\n              DW_AT_low_pc (0x1000000000000000)\n              DW_AT_high_pc (0x1000000000000010)'
validate_concrete_subprograms parser-control 1 1 <<<"${large_valid_sample}" >/dev/null 2>&1 \
    || fail "internal parser rejected its 64-bit valid control"
invalid_samples=(
    ""
    $'0x00000001: DW_TAG_subprogram\n              DW_AT_name ("abstract")'
    $'0x00000001: DW_TAG_inlined_subroutine\n              DW_AT_low_pc (0x0)\n              DW_AT_high_pc (0x10)'
    $'0x00000001: DW_TAG_subprogram\n              DW_AT_ranges (0x0)\n              DW_AT_low_pc (0x0)\n              DW_AT_high_pc (0x10)'
    $'0x00000001: DW_TAG_subprogram\n              DW_AT_low_pc (not-a-pc)\n              DW_AT_high_pc (0x10)'
    $'0x00000001: DW_TAG_subprogram\n              DW_AT_low_pc (0x10)\n              DW_AT_high_pc (0x10)'
    $'0x00000001: DW_TAG_subprogram\n              DW_AT_low_pc (0x20)\n              DW_AT_high_pc (0x10)'
)
for invalid_sample in "${invalid_samples[@]}"; do
    if validate_concrete_subprograms parser-control 1 1 <<<"${invalid_sample}" >/dev/null 2>&1; then
        fail "internal parser accepted an invalid control"
    fi
done

tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT
cubin="${tmpdir}/compiler_features.cubin"
get_mut_dump="${tmpdir}/get-mut.dwarf"
caller_dump="${tmpdir}/test-option.dwarf"

"${ptxas_bin}" -arch="${arch}" -g "${ptx}" -o "${cubin}" \
    || fail "ptxas rejected the full-debug PTX for ${arch}"
[[ -s "${cubin}" ]] || fail "ptxas emitted an empty cubin"

# Do not add a blanket `llvm-dwarfdump --verify` here. CUDA cubins place
# functions in distinct text sections whose valid low_pc values can all be
# zero; llvm-dwarfdump 20/21 treats those section-relative ranges as
# overlapping and exits nonzero. Successful ptxas assembly, targeted DWARF
# decoding, and the per-DIE contracts below avoid that known false positive.
"${dwarfdump_bin}" --debug-info --regex \
    --name '.*DisjointSlice.*::get_mut$' "${cubin}" >"${get_mut_dump}" \
    || fail "llvm-dwarfdump could not query get_mut"
if ! validate_concrete_subprograms get_mut 1 0 <"${get_mut_dump}"; then
    fail "get_mut is not represented solely by concrete contiguous subprogram DIEs"
fi

"${dwarfdump_bin}" --debug-info --name test_option "${cubin}" >"${caller_dump}" \
    || fail "llvm-dwarfdump could not query test_option"
if ! validate_concrete_subprograms test_option 1 1 <"${caller_dump}"; then
    fail "test_option is not one concrete contiguous kernel subprogram DIE"
fi

echo "compiler-features debug-info shape verified (physical get_mut and test_option ranges; arch: ${arch})"
