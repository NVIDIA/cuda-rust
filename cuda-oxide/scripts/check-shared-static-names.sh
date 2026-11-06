#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Verify that adding a kernel leaves every other kernel's PTX unchanged (#1477).
#
# A `SharedArray` static lowers to a `.shared` global and an ordinary device
# `static` to a `.global`. Both used to be named from a per-module counter in
# MIR walk order (`__shared_mem_<crate>_<n>`, `__device_global_<crate>_<n>`),
# so adding a kernel renumbered the other kernels' globals and changed their
# PTX and cubins although their code had not changed. Release builds also keyed
# shared statics by the constant's Debug spelling rather than by the static.
# Lowering now names each global after a hash of the static's mangled symbol,
# plus the field's path for a pointer to a `SharedArray` field of a tuple
# static, in every build profile.
#
# The check builds the shared_static_names example, then two copies of it with
# added kernels, each carrying its own shared and device-global statics,
# spliced in at the `ADDED KERNEL SLOT (first)` or `(last)` comment. Each copy
# keeps the package name, so the crate component of the names is unchanged, and
# the kernels are spliced in on that comment's line, so full-debug line numbers
# stay put. For a release build and a full-debug (`--device-debug`) build, the
# original kernels' `.entry` bodies and the declarations of every global they
# reference must be identical between the original and each copy.
#
# Lowering walks kernels in rustc's mono-item order, which follows symbol
# hashes rather than source order, so a single added kernel can land after the
# originals and shift nothing even under counter naming. Several are added, and
# the check fails its own self-test unless lowering reached at least one added
# shared static before an original one, which it reads from the declaration
# order: lowering inserts each shared global at the front of the module.
#
# Position-dependent output from LLVM's PTX printer is outside this contract.
# It prints a `// @<name>` comment before the module's first function and
# after every other function's parameter list, so the comparison drops those
# comments. It numbers basic-block labels by the function's position in the
# module (`$L__BB<function>_<block>`), so the comparison drops the function
# number and keeps the block number; a release build must match byte for byte
# otherwise. A full-debug build also numbers its `$L__tmp` and `$L__func_*`
# labels and its `__local_depot` stack frames across the whole module, so there
# all labels and depots are renumbered in first-use order within each kernel
# before comparing; the global names must still match exactly.
#
# The copies live next to the example as `.shared_static_names_<slot>` so its
# relative path dependencies resolve, and are removed on exit.
#
# Needs what `cargo oxide build` needs (the backend toolchain and CUDA
# toolkit), but no GPU. Set SHARED_STATIC_NAMES_ARCH to change the target
# (default sm_80).
#
# Run this after changing how shared or device-global symbols are named.
set -euo pipefail

export LC_ALL=C

cd "$(dirname "$0")/.."

EXAMPLES=crates/rustc-codegen-cuda/examples
EXAMPLE=shared_static_names
ARCH="${SHARED_STATIC_NAMES_ARCH:-sm_80}"
KERNELS=(rotate_128 rotate_256 field_tiles)
# field_tiles points at four `SharedArray` fields, each its own global, and
# reads one back through a `*const` pointer that must share its global
FIELD_KERNEL=field_tiles
FIELD_GLOBALS=4
ADDED_KERNELS=(added_a added_b added_c added_d added_e added_f)
# all on one line, so splicing them over a slot comment moves no other source
# line
ADDED_SOURCE="   "
for kernel in "${ADDED_KERNELS[@]}"; do
    ADDED_SOURCE+=" #[kernel] pub fn ${kernel}(mut out: DisjointSlice<f32>) { static mut TILE: SharedArray<f32, 64, 16> = SharedArray::UNINIT; static OFFSETS: [f32; 2] = [0.5, 1.5]; let tile = unsafe { SharedArray::as_raw_mut_ptr(&raw mut TILE) }; let tid = thread::threadIdx_x() as usize; unsafe { *tile.add(tid & 63) = tid as f32 }; thread::sync_threads(); unsafe { *out.get_unchecked_mut(tid) = *tile.add((tid + 1) & 63) + OFFSETS[tid & 1] }; }"
done

if ! command -v python3 >/dev/null 2>&1; then
    echo "error: python3 is required to compare the PTX" >&2
    echo "       refusing to report success from a check that cannot run" >&2
    exit 1
fi

work="$(mktemp -d)"
variants=()
cleanup() {
    rm -rf "${work}"
    if [ "${#variants[@]}" -gt 0 ]; then
        rm -rf "${variants[@]}"
    fi
}
trap cleanup EXIT

# build <example dir name> <label> [cargo oxide build flags...]
build() {
    local name="$1"
    local label="$2"
    shift 2
    local log="${work}/${label}.log"
    local ptx="${EXAMPLES}/${name}/${EXAMPLE}.ptx"
    # the copies share the example's package id, so with a shared
    # CARGO_TARGET_DIR a build can be fresh; never compare a leftover PTX
    rm -f "${ptx}"
    if ! cargo oxide build "${name}" --arch "${ARCH}" "$@" >"${log}" 2>&1; then
        cat "${log}" >&2
        echo "error: cargo oxide build ${name} $* failed" >&2
        exit 1
    fi
    if [ ! -s "${ptx}" ]; then
        cat "${log}" >&2
        echo "error: cargo oxide build ${name} $* wrote no ${ptx}" >&2
        exit 1
    fi
    cp "${ptx}" "${work}/${label}.ptx"
}

for slot in first last; do
    variant="${EXAMPLES}/.${EXAMPLE}_${slot}"
    rm -rf "${variant}"
    variants+=("${variant}")
    mkdir "${variant}"
    cp -R "${EXAMPLES}/${EXAMPLE}/Cargo.toml" "${EXAMPLES}/${EXAMPLE}/Cargo.lock" \
        "${EXAMPLES}/${EXAMPLE}/src" "${variant}/"
    ADDED_SOURCE="${ADDED_SOURCE}" python3 - "${variant}/src/main.rs" "${slot}" <<'EOF'
import os
import sys

path, slot = sys.argv[1], sys.argv[2]
marker = f"    // ADDED KERNEL SLOT ({slot})\n"
source = open(path).read()
if source.count(marker) != 1:
    sys.exit(f"error: {path} must contain exactly one `{marker.strip()}` line")
open(path, "w").write(source.replace(marker, os.environ["ADDED_SOURCE"] + "\n"))
EOF
done

for mode in release full-debug; do
    flags=()
    if [ "${mode}" = full-debug ]; then
        flags=(--device-debug)
    fi
    build "${EXAMPLE}" "${mode}-base" "${flags[@]}"
    for slot in first last; do
        build ".${EXAMPLE}_${slot}" "${mode}-${slot}" "${flags[@]}"
        python3 - "${work}/${mode}-base.ptx" "${work}/${mode}-${slot}.ptx" \
            "${mode}" "${slot}" "${FIELD_KERNEL}" "${FIELD_GLOBALS}" \
            "${ADDED_KERNELS[*]}" "${KERNELS[@]}" <<'EOF'
import difflib
import re
import sys

base_path, variant_path, mode, slot, field_kernel, field_globals, added, *kernels = sys.argv[1:]
added = added.split()
label = f"{mode}, kernels added {slot}"
GLOBAL = re.compile(r"__(?:shared_mem|device_global)_[0-9a-f_]+")
LABEL = re.compile(r"\$L__\w+|__local_depot\d+")
BLOCK_FUNCTION = re.compile(r"\$L__BB\d+_")


def canonical_labels(body):
    """Renumber LLVM's module-wide labels and stack depots in first-use order."""
    names = {}
    return LABEL.sub(lambda match: names.setdefault(match.group(0), f"$L__{len(names)}"), body)


def parse(path):
    """Return each `.entry` body and each module-scope global's declaration."""
    lines = open(path).read().split("\n")
    entries, declarations, order = {}, {}, []
    index = 0
    while index < len(lines):
        line = lines[index]
        entry = re.match(r"\.visible \.entry (\w+)\(", line)
        if entry:
            end = lines.index("}", index)
            # the printer places this comment by module position alone
            body = (re.sub(r"\s*// @\w+$", "", body_line) for body_line in lines[index : end + 1])
            body = "\n".join(body)
            if mode == "full-debug":
                body = canonical_labels(body)
            else:
                body = BLOCK_FUNCTION.sub("$L__BB_", body)
            entries[entry.group(1)] = body
            index = end
        else:
            # any linkage and either form: `name[size]` or a scalar `name;`
            declared = re.match(
                r"(?:\.(?:visible|extern|weak|common)\s+)*\.(?:shared|global|const)\s.*?("
                + GLOBAL.pattern
                + r")\s*(?:\[|;|=)",
                line,
            )
            if declared:
                declarations[declared.group(1)] = line
                order.append(declared.group(1))
        index += 1
    return entries, declarations, order


def referenced(body):
    return sorted(set(GLOBAL.findall(body)))


base_entries, base_declarations, _ = parse(base_path)
variant_entries, variant_declarations, variant_order = parse(variant_path)

# self-tests: without these the comparison below could pass vacuously
for kernel in kernels:
    if kernel not in base_entries:
        sys.exit(f"parse self-test failed: no `.entry {kernel}` in {base_path}")
    if not any(name.startswith("__shared_mem_") for name in referenced(base_entries[kernel])):
        sys.exit(f"parse self-test failed: {kernel} references no __shared_mem_* global")
if not any(
    name.startswith("__device_global_")
    for kernel in kernels
    for name in referenced(base_entries[kernel])
):
    sys.exit("parse self-test failed: no kernel references a __device_global_* global")
field_shared = [
    name for name in referenced(base_entries[field_kernel]) if name.startswith("__shared_mem_")
]
if len(field_shared) != int(field_globals):
    sys.exit(
        f"field self-test failed: {field_kernel} references {len(field_shared)} shared"
        f" globals, expected {field_globals}, one per `SharedArray` field"
    )
added_globals = set()
for kernel in added:
    if kernel not in variant_entries:
        sys.exit(f"parse self-test failed: no `.entry {kernel}` in {variant_path}")
    globals_ = set(referenced(variant_entries[kernel]))
    if not any(name.startswith("__shared_mem_") for name in globals_) or not any(
        name.startswith("__device_global_") for name in globals_
    ):
        sys.exit(f"parse self-test failed: {kernel} references no shared and device globals")
    added_globals |= globals_
original_globals = {
    name for kernel in kernels for name in referenced(variant_entries.get(kernel, ""))
}

# shared globals are inserted at the front of the module, so the last one
# declared is the first one lowering reached
walk = [name for name in reversed(variant_order) if name.startswith("__shared_mem_")]
first_added = min(walk.index(name) for name in added_globals if name in walk)
last_original = max(walk.index(name) for name in original_globals if name in walk)
if first_added > last_original:
    sys.exit(
        "sensitivity self-test failed: lowering reached every added shared static"
        " after the original ones, so a walk-order counter would also pass;"
        " rename or add kernels in ADDED_KERNELS"
    )

failures = []
for kernel in kernels:
    if variant_entries.get(kernel) != base_entries[kernel]:
        diff = difflib.unified_diff(
            base_entries[kernel].split("\n"),
            variant_entries.get(kernel, "").split("\n"),
            "original",
            "kernels added",
            lineterm="",
        )
        failures.append(f"`.entry {kernel}` body changed:\n    " + "\n    ".join(diff))
    for name in referenced(base_entries[kernel]):
        # a global with no parsed declaration on either side would compare
        # equal as missing, so require one before comparing
        if name not in base_declarations:
            sys.exit(f"parse self-test failed: no declaration of {name} in {base_path}")
        if variant_declarations.get(name) != base_declarations[name]:
            failures.append(f"{kernel}: declaration of {name} changed or disappeared")
if failures:
    print(f"error ({label}): adding a kernel changed other kernels' PTX (#1477):", file=sys.stderr)
    for failure in failures:
        print(f"  - {failure}", file=sys.stderr)
    sys.exit(1)
print(f"OK ({label}): {', '.join(kernels)} PTX unchanged")
EOF
    done
done
