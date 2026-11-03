#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Verify the structural half of the raw-pointer-source-local debug contract. The
# cuda-gdb half lives in scripts/debug-smoketest.sh; both are needed because a
# well-typed DILocalVariable can still point at an alloca that is never written.

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
llvm_ir="${1:-${root}/debug_pointer_locals.ll}"

fail() {
    echo "raw-pointer debug-info shape: FAIL ($1)" >&2
    exit 1
}

[[ -s "${llvm_ir}" ]] || fail "missing LLVM IR at ${llvm_ir}"

kernel_body="$(awk '
    /^define ptx_kernel void @debug_pointer_locals\(/ { in_function = 1 }
    in_function { print }
    in_function && /^}/ { exit }
' "${llvm_ir}")"
[[ -n "${kernel_body}" ]] || fail "debug_pointer_locals kernel definition not found"

slots=()

block_at_line() {
    local line="$1"
    head -n "${line}" <<<"${kernel_body}" | grep -E '^[[:alnum:]_.-]+:$' | tail -1
}

cfg_edges="$(awk '
    /^[[:alnum:]_.-]+:$/ {
        block = $0
        sub(/:$/, "", block)
    }
    {
        rest = $0
        while (match(rest, /label %[[:alnum:]_.-]+/)) {
            target = substr(rest, RSTART + 7, RLENGTH - 7)
            print block, target
            rest = substr(rest, RSTART + RLENGTH)
        }
    }
' <<<"${kernel_body}")"

# Return success when every CFG path from entry to target passes dominator.
# SSA verification does not prove this property for values flowing through
# memory, so the debug-slot gate checks it explicitly.
block_dominates() {
    local dominator="${1%:}"
    local target="${2%:}"
    local current successor
    local -a queue=(entry)
    local -A seen=([entry]=1)

    [[ "${dominator}" == "${target}" || "${dominator}" == "entry" ]] && return 0
    while [[ "${#queue[@]}" -gt 0 ]]; do
        current="${queue[0]}"
        queue=("${queue[@]:1}")
        while IFS= read -r successor; do
            [[ -n "${successor}" && "${successor}" != "${dominator}" ]] || continue
            [[ "${successor}" != "${target}" ]] || return 1
            if [[ -z "${seen[${successor}]+x}" ]]; then
                seen["${successor}"]=1
                queue+=("${successor}")
            fi
        done < <(awk -v block="${current}" '$1 == block { print $2 }' <<<"${cfg_edges}")
    done
    return 0
}

# Require pointer/aggregate backing storage to be used directly. This fixture
# does not need address casts or GEP aliases for its debug temporaries; allowing
# them would let an intervening aliasing store invalidate the direct reaching-
# store proof below. The one optional address-forward line is used only while
# resolving ThreadIndex::get's source alloca.
require_direct_storage_uses() {
    local source_name="$1"
    local storage="$2"
    local allowed_forward_line="${3:-}"
    local uses line line_number line_text

    uses="$(grep -nE "(^|[^[:alnum:]_.])${storage}([^[:alnum:]_.]|$)" \
        <<<"${kernel_body}" || true)"
    while IFS= read -r line; do
        [[ -n "${line}" ]] || continue
        line_number="${line%%:*}"
        line_text="${line#*:}"
        if [[ "${line_text}" == "  ${storage} = alloca "* \
            || "${line_text}" == *"@llvm.dbg.declare(metadata ptr ${storage},"* \
            || "${line_text}" =~ ^[[:space:]]*store\ .+,\ ptr\ ${storage}, \
            || "${line_text}" =~ ^[[:space:]]*%[^[:space:]]+[[:space:]]=[[:space:]]load\ .+,\ ptr\ ${storage}, ]]; then
            continue
        fi
        if [[ -n "${allowed_forward_line}" \
            && "${line_number}" -eq "${allowed_forward_line}" \
            && "${line_text}" =~ ^[[:space:]]*store\ ptr\ ${storage},\ ptr\  ]]; then
            continue
        fi
        fail "${source_name} backing storage ${storage} has aliasing/unsupported use: ${line_text}"
    done <<<"${uses}"
}

# The allowlisted ThreadIndex::get call dereferences its pointer argument.
# Resolve the fixture's pointer-forwarding loads to the source alloca and prove
# that alloca has an initialized, dominating value before the call.
require_initialized_pointee() {
    local source_name="$1"
    local operand="$2"
    local use_line="$3"
    local definition definition_line definition_text backing stores store
    local store_line store_text store_prefix store_operand definition_block store_block

    [[ "${operand}" == %* ]] \
        || fail "${source_name} accessor pointee is not SSA storage: ${operand}"
    definition="$(grep -nE "^  ${operand} = " <<<"${kernel_body}" || true)"
    [[ "$(grep -c . <<<"${definition}" || true)" -eq 1 ]] \
        || fail "${source_name} accessor storage ${operand} is missing or ambiguous"
    definition_line="${definition%%:*}"
    definition_text="${definition#*:}"
    [[ "${definition_line}" -lt "${use_line}" ]] \
        || fail "${source_name} accessor storage does not dominate its use"

    if [[ "${definition_text}" =~ load\ ptr,\ ptr\ (%[^,]+), ]]; then
        backing="${BASH_REMATCH[1]}"
        require_direct_storage_uses "${source_name}" "${backing}"
        stores="$(grep -nE "store ptr [^,]+, ptr ${backing}," \
            <<<"${kernel_body}" | awk -F: -v limit="${definition_line}" '$1 < limit' || true)"
        [[ -n "${stores}" ]] \
            || fail "${source_name} accessor pointer loads from uninitialized ${backing}"
        store="$(tail -1 <<<"${stores}")"
        store_line="${store%%:*}"
        store_text="${store#*:}"
        definition_block="$(block_at_line "${definition_line}")"
        store_block="$(block_at_line "${store_line}")"
        block_dominates "${store_block}" "${definition_block}" \
            || fail "${source_name} accessor pointer store does not dominate its load"
        if [[ "${store_block}" == "${definition_block}" ]]; then
            [[ "$((store_line + 1))" -eq "${definition_line}" ]] \
                || fail "${source_name} accessor pointer store/load is not adjacent"
        fi
        store_prefix="${store_text%%, ptr ${backing},*}"
        store_operand="${store_prefix##* }"
        require_initialized_pointee "${source_name}" "${store_operand}" "${store_line}"
        return
    fi

    if [[ "${definition_text}" =~ alloca\  ]]; then
        require_direct_storage_uses "${source_name}" "${operand}" "${use_line}"
        stores="$(grep -nE "store .+, ptr ${operand}," \
            <<<"${kernel_body}" | awk -F: -v limit="${use_line}" '$1 < limit' || true)"
        [[ -n "${stores}" ]] \
            || fail "${source_name} accessor pointee ${operand} is never initialized"
        store="$(tail -1 <<<"${stores}")"
        store_line="${store%%:*}"
        store_text="${store#*:}"
        definition_block="$(block_at_line "${use_line}")"
        store_block="$(block_at_line "${store_line}")"
        block_dominates "${store_block}" "${definition_block}" \
            || fail "${source_name} accessor pointee store does not dominate its call"
        [[ "${store_text}" != *poison* && "${store_text}" != *undef* ]] \
            || fail "${source_name} accessor pointee contains poison/undef"
        store_prefix="${store_text%%, ptr ${operand},*}"
        store_operand="${store_prefix##* }"
        require_initialized_producer "${source_name}" "${store_operand}" "${store_line}"
        return
    fi

    fail "${source_name} accessor has unsupported pointee source: ${definition_text}"
}

# Follow the bounded producer shapes this fixture intentionally emits. A store
# into the final dbg.declare slot is not enough: it could copy poison from an
# uninitialized temporary. Loads must have an earlier producer store in the
# same block or in the kernel's unconditional bb0 prologue. Every accepted SSA
# opcode is handled explicitly and recursively; an unfamiliar opcode is a hard
# failure so select/phi/call wrappers cannot hide poison from this gate.
require_initialized_producer() {
    local source_name="$1"
    local operand="$2"
    local use_line="$3"
    local projection="${4:-}"
    local definition definition_line definition_text inner backing stores store
    local store_line store_text store_prefix store_operand definition_block store_block
    local base inserted inserted_index aggregate_index incoming incoming_operand
    local gep_base gep_index call_argument
    local thread_index_call_re='call i64 @[^[:space:]]*ThreadIndex3get[^[:space:]]*debug_pointer_locals\(ptr (%[^)]+)\)'

    [[ "${operand}" != *poison* && "${operand}" != *undef* ]] \
        || fail "${source_name} debug value is poison/undef"

    # Literal scalars and null are initialized values. Kernel parameters are
    # also roots, but only when they occur in this kernel's formal list.
    if [[ "${operand}" == "null" || "${operand}" =~ ^-?[0-9]+$ \
        || "${operand}" == "true" || "${operand}" == "false" ]]; then
        [[ -z "${projection}" ]] \
            || fail "${source_name} projects field ${projection} from scalar ${operand}"
        return
    fi
    [[ "${operand}" == %* ]] \
        || fail "${source_name} debug value is not initialized: ${operand}"
    if head -1 <<<"${kernel_body}" \
        | grep -Eq "[(,][[:space:]]*[^,%]+ ${operand}([,)])"; then
        [[ -z "${projection}" ]] \
            || fail "${source_name} projects field ${projection} from scalar kernel argument ${operand}"
        return
    fi

    definition="$(grep -nE "^  ${operand} = " <<<"${kernel_body}" || true)"
    [[ "$(grep -c . <<<"${definition}" || true)" -eq 1 ]] \
        || fail "${source_name} SSA producer ${operand} is missing or ambiguous"
    definition_line="${definition%%:*}"
    definition_text="${definition#*:}"
    [[ "${definition_line}" -lt "${use_line}" ]] \
        || fail "${source_name} SSA producer does not dominate its use"

    if [[ "${definition_text}" =~ inttoptr\ i64\ (%[^[:space:]]+)\ to\ ptr ]]; then
        [[ -z "${projection}" ]] \
            || fail "${source_name} projects field ${projection} from inttoptr"
        inner="${BASH_REMATCH[1]}"
        require_initialized_producer "${source_name}" "${inner}" "${definition_line}"
        return
    fi

    if [[ "${definition_text}" =~ load\ .*,\ ptr\ (%[^,]+), ]]; then
        backing="${BASH_REMATCH[1]}"
        if [[ "${definition_text}" == *" = load ptr,"* \
            || "${definition_text}" == *" = load {"* ]]; then
            require_direct_storage_uses "${source_name}" "${backing}"
        fi
        stores="$(grep -nE "store .+, ptr ${backing}," \
            <<<"${kernel_body}" | awk -F: -v limit="${definition_line}" '$1 < limit' || true)"
        [[ -n "${stores}" ]] \
            || fail "${source_name} SSA producer loads from uninitialized ${backing}"
        store="$(tail -1 <<<"${stores}")"
        store_line="${store%%:*}"
        store_text="${store#*:}"
        [[ "${store_text}" != *poison* && "${store_text}" != *undef* ]] \
            || fail "${source_name} backing storage contains poison/undef"
        definition_block="$(block_at_line "${definition_line}")"
        store_block="$(block_at_line "${store_line}")"
        block_dominates "${store_block}" "${definition_block}" \
            || fail "${source_name} backing store does not dominate its load"
        if [[ "${store_block}" == "${definition_block}" ]]; then
            [[ "$((store_line + 1))" -eq "${definition_line}" ]] \
                || fail "${source_name} same-block producer store/load is not adjacent"
        fi
        store_prefix="${store_text%%, ptr ${backing},*}"
        store_operand="${store_prefix##* }"
        require_initialized_producer \
            "${source_name}" "${store_operand}" "${store_line}" "${projection}"
        return
    fi

    if [[ "${definition_text}" =~ getelementptr.*ptr\ (%[^,[:space:]]+) ]]; then
        [[ -z "${projection}" ]] \
            || fail "${source_name} projects field ${projection} from getelementptr"
        [[ "${definition_text}" != *poison* && "${definition_text}" != *undef* ]] \
            || fail "${source_name} getelementptr contains poison/undef"
        gep_base="${BASH_REMATCH[1]}"
        require_initialized_producer "${source_name}" "${gep_base}" "${definition_line}"
        while IFS= read -r gep_index; do
            [[ -n "${gep_index}" ]] || continue
            require_initialized_producer "${source_name}" "${gep_index}" "${definition_line}"
        done < <(grep -oE ', i[0-9]+ %[^,[:space:]]+' <<<"${definition_text}" \
            | sed -E 's/.*, i[0-9]+ //')
        return
    fi

    if [[ "${definition_text}" =~ extractvalue.*\ (%[^,[:space:]]+),\ ([0-9]+) ]]; then
        [[ -z "${projection}" ]] \
            || fail "${source_name} applies nested projection to extractvalue"
        inner="${BASH_REMATCH[1]}"
        aggregate_index="${BASH_REMATCH[2]}"
        require_initialized_producer \
            "${source_name}" "${inner}" "${definition_line}" "${aggregate_index}"
        return
    fi

    if [[ "${definition_text}" =~ insertvalue.*[[:space:]]([^[:space:],]+),[[:space:]][^[:space:]]+[[:space:]]([^[:space:],]+),[[:space:]]([0-9]+) ]]; then
        base="${BASH_REMATCH[1]}"
        inserted="${BASH_REMATCH[2]}"
        inserted_index="${BASH_REMATCH[3]}"
        [[ -n "${projection}" ]] \
            || fail "${source_name} consumes insertvalue without a projection"
        if [[ "${projection}" == "${inserted_index}" ]]; then
            require_initialized_producer \
                "${source_name}" "${inserted}" "${definition_line}"
        else
            require_initialized_producer \
                "${source_name}" "${base}" "${definition_line}" "${projection}"
        fi
        return
    fi

    if [[ "${definition_text}" =~ phi\  ]]; then
        incoming="$(grep -oE '\[ ([^,]+), %[^]]+ \]' <<<"${definition_text}" || true)"
        [[ -n "${incoming}" ]] || fail "${source_name} has malformed phi producer"
        while IFS= read -r incoming; do
            incoming_operand="$(sed -E 's/^\[ ([^,]+),.*/\1/' <<<"${incoming}")"
            require_initialized_producer \
                "${source_name}" "${incoming_operand}" "${definition_line}" "${projection}"
        done <<<"${incoming}"
        return
    fi

    # The fixture's GEP index ultimately comes from this one semantically
    # noundef accessor. Keep the allowlist exact: arbitrary calls remain a hard
    # failure, while the accessor's pointer argument is still traced.
    if [[ "${definition_text}" =~ ${thread_index_call_re} ]]; then
        [[ -z "${projection}" ]] \
            || fail "${source_name} projects field ${projection} from ThreadIndex::get"
        [[ "${definition_text}" != *poison* && "${definition_text}" != *undef* ]] \
            || fail "${source_name} ThreadIndex::get contains poison/undef"
        call_argument="${BASH_REMATCH[1]}"
        require_initialized_pointee \
            "${source_name}" "${call_argument}" "${definition_line}"
        return
    fi

    if [[ "${definition_text}" =~ call\ i64\ @[^[:space:]]*index_1d[^[:space:]]*debug_pointer_locals\( ]]; then
        [[ -z "${projection}" ]] \
            || fail "${source_name} projects field ${projection} from index_1d"
        [[ "${definition_text}" != *poison* && "${definition_text}" != *undef* ]] \
            || fail "${source_name} index_1d contains poison/undef"
        return
    fi

    fail "${source_name} uses unsupported SSA producer: ${definition_text}"
}

require_pointer_local() {
    local source_name="$1"
    local expected_type="$2"
    local expected_base="$3"
    local die die_id type_id type_die base_id base_die declare slot stores store_count
    local debug_store debug_store_line debug_store_text debug_operand

    die="$(grep -F "!DILocalVariable(name: \"${source_name}\"" "${llvm_ir}" || true)"
    [[ -n "${die}" ]] || fail "${source_name} DILocalVariable missing"
    [[ "$(printf '%s\n' "${die}" | wc -l)" -eq 1 ]] \
        || fail "${source_name} has more than one DILocalVariable"
    die_id="$(sed -nE 's/^!([0-9]+) = .*/\1/p' <<<"${die}")"
    type_id="$(sed -nE 's/.*type: !([0-9]+).*/\1/p' <<<"${die}")"
    [[ -n "${die_id}" && -n "${type_id}" ]] || fail "${source_name} metadata is malformed"

    type_die="$(grep -E "^!${type_id} = !DIDerivedType\\(" "${llvm_ir}" || true)"
    [[ -n "${type_die}" ]] || fail "${source_name} pointer type metadata missing"
    grep -Fq 'tag: DW_TAG_pointer_type' <<<"${type_die}" \
        || fail "${source_name} is not described as a pointer"
    grep -Fq "name: \"${expected_type}\"" <<<"${type_die}" \
        || fail "${source_name} lost source type ${expected_type}"
    base_id="$(sed -nE 's/.*baseType: !([0-9]+).*/\1/p' <<<"${type_die}")"
    base_die="$(grep -E "^!${base_id} = !DIBasicType\\(" "${llvm_ir}" || true)"
    grep -Fq "name: \"${expected_base}\"" <<<"${base_die}" \
        || fail "${source_name} lost pointee type ${expected_base}"

    declare="$(grep -F "metadata !${die_id}," <<<"${kernel_body}" \
        | grep -F '@llvm.dbg.declare' || true)"
    [[ -n "${declare}" ]] || fail "${source_name} has no llvm.dbg.declare"
    [[ "$(printf '%s\n' "${declare}" | wc -l)" -eq 1 ]] \
        || fail "${source_name} has more than one llvm.dbg.declare"
    slot="$(sed -nE 's/.*metadata ptr (%[^,]+), metadata.*/\1/p' <<<"${declare}")"
    [[ -n "${slot}" ]] || fail "${source_name} debug storage cannot be resolved"

    stores="$(grep -E "store ptr [^,]+, ptr ${slot}," <<<"${kernel_body}" || true)"
    store_count="$(grep -c . <<<"${stores}" || true)"
    [[ "${store_count}" -ge 1 ]] \
        || fail "${source_name} debug storage is never initialized"
    ! grep -Eq '(^|[^[:alnum:]_])(poison|undef)([^[:alnum:]_]|$)' <<<"${stores}" \
        || fail "${source_name} debug storage is initialized with poison/undef"
    debug_store="$(grep -nE "store ptr [^,]+, ptr ${slot}," <<<"${kernel_body}")"
    [[ "$(grep -c . <<<"${debug_store}" || true)" -eq 1 ]] \
        || fail "${source_name} must have one unambiguous debug-slot store"
    debug_store_line="${debug_store%%:*}"
    debug_store_text="${debug_store#*:}"
    debug_operand="$(sed -nE "s/.*store ptr ([^,]+), ptr ${slot},.*/\1/p" \
        <<<"${debug_store_text}")"
    [[ -n "${debug_operand}" ]] || fail "${source_name} debug-slot value is malformed"
    require_initialized_producer \
        "${source_name}" "${debug_operand}" "${debug_store_line}"
    ! grep -Eq "@llvm[.]dbg[.]value\\(metadata ptr (poison|undef), metadata !${die_id}," \
        <<<"${kernel_body}" || fail "${source_name} is marked optimized out"

    slots+=("${slot}")
}

# ReferencePropagation reuses the reference-producing MIR locals for these two
# source bindings. Match pinned native rustc and require their post-pass Rust
# debug types, rather than fabricating the pre-pass raw-pointer spelling.
require_pointer_local ptr '&i32' i32
require_pointer_local fptr '&f32' f32
require_pointer_local null_ptr '*const i32' i32

[[ "${slots[0]}" != "${slots[1]}" && "${slots[0]}" != "${slots[2]}" \
    && "${slots[1]}" != "${slots[2]}" ]] \
    || fail "pointer locals unexpectedly share one debug slot"

# Verify with complete, version-matched LLVM tool pairs. LLVM 18 can return
# success while diagnosing and stripping newer debug metadata, so accepting an
# arbitrary unsuffixed llvm-as would turn this gate into a false assurance.
# CUDA Oxide's exported debug graph is supported and tested with LLVM 21+.
tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT

toolsets=()
if command -v rustc >/dev/null 2>&1; then
    sysroot_bin="$(rustc --print sysroot)/lib/rustlib/$(rustc -vV | sed -n 's/^host: //p')/bin"
    if [[ -x "${sysroot_bin}/llvm-as" && -x "${sysroot_bin}/opt" ]]; then
        toolsets+=("${sysroot_bin}/llvm-as|${sysroot_bin}/opt")
    fi
fi
for version in 21 22 23 24 25; do
    if command -v "llvm-as-${version}" >/dev/null 2>&1 \
        && command -v "opt-${version}" >/dev/null 2>&1; then
        toolsets+=("$(command -v "llvm-as-${version}")|$(command -v "opt-${version}")")
    fi
done
if command -v llvm-as >/dev/null 2>&1 && command -v opt >/dev/null 2>&1; then
    toolsets+=("$(command -v llvm-as)|$(command -v opt)")
fi

validated=0
declare -A seen_llvm_majors=()
for entry in "${toolsets[@]}"; do
    llvm_as="${entry%%|*}"
    opt="${entry##*|}"
    llvm_as_major="$("${llvm_as}" --version \
        | sed -nE 's/.*LLVM version ([0-9]+).*/\1/p' | head -n 1)"
    opt_major="$("${opt}" --version \
        | sed -nE 's/.*LLVM version ([0-9]+).*/\1/p' | head -n 1)"
    [[ "${llvm_as_major}" =~ ^[0-9]+$ && "${opt_major}" =~ ^[0-9]+$ ]] || continue
    [[ "${llvm_as_major}" -eq "${opt_major}" && "${llvm_as_major}" -ge 21 ]] || continue
    [[ -z "${seen_llvm_majors[${llvm_as_major}]+x}" ]] || continue
    seen_llvm_majors["${llvm_as_major}"]=1

    bitcode="${tmpdir}/debug-pointer-locals-${llvm_as_major}.bc"
    as_err="${tmpdir}/llvm-as-${llvm_as_major}.err"
    opt_err="${tmpdir}/opt-${llvm_as_major}.err"
    if ! "${llvm_as}" -o "${bitcode}" "${llvm_ir}" 2>"${as_err}"; then
        cat "${as_err}" >&2
        fail "${llvm_as} could not assemble the LLVM IR"
    fi
    [[ -s "${bitcode}" ]] || fail "${llvm_as} produced no bitcode"
    if grep -qi 'invalid debug info' "${as_err}"; then
        cat "${as_err}" >&2
        fail "${llvm_as} stripped invalid debug metadata"
    fi
    if ! "${opt}" -passes=verify -disable-output "${bitcode}" 2>"${opt_err}"; then
        cat "${opt_err}" >&2
        fail "${opt} rejected the LLVM IR"
    fi
    if grep -qi 'invalid debug info' "${opt_err}"; then
        cat "${opt_err}" >&2
        fail "${opt} stripped invalid debug metadata"
    fi
    validated=$((validated + 1))
done

[[ "${validated}" -gt 0 ]] \
    || fail "no complete version-matched LLVM 21+ llvm-as/opt pair is available"

echo "raw-pointer debug-info shape: PASS"
