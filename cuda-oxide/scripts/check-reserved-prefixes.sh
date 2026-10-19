#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Reject any Rust source line, outside the canonical `reserved-oxide-symbols`
# crate, that hardcodes a reserved *symbol* prefix as a string literal.
#
# The single source of truth for these prefixes is
# `cuda-oxide/crates/reserved-oxide-symbols/`; everything else must route
# through that crate's constants, builders, and predicates.
#
# The alternation below covers all ten symbol constants that crate exports:
# `kernel` covers LEGACY_KERNEL_PREFIX and KERNEL_SCOPE_LOCAL; `device` covers
# LEGACY_DEVICE_PREFIX and DEVICE_EXTERN_PREFIX; `instantiate`, `const`,
# `artifact_anchor` and `ptx_merge_required` cover one each; and `codegen_v1`
# covers the modern KERNEL_PREFIX and DEVICE_PREFIX.  The last one matters
# because the pattern is anchored to the opening quote: a hardcoded full modern
# prefix starts with `cuda_oxide_codegen_v1_`, so its embedded
# `cuda_oxide_kernel_` segment sits mid-string where the quote-anchored `kernel`
# family can never match it.
#
# It deliberately does NOT match bare `cuda_oxide_`.  That word-space also holds
# things this crate does not own and must not police: pliron op-attribute keys
# (`cuda_oxide_debug_local_*`), rustc `--cfg` names
# (`cuda_oxide_internal_backend_identity`), dlopen probe symbols
# (`cuda_oxide_probe`), and unique temp-directory names.  Those are legitimate
# local constants, not reserved link symbols.
#
# The pipeline is three steps:
#   1. a self-test proves the search still matches a known violation;
#   2. grep finds candidate lines containing `"cuda_oxide_*_`;
#   3. grep -v drops the canonical crate, then drops lines whose content (after
#      `file:lineno:`) starts with optional whitespace then `//` -- pure comment
#      lines, which are free to mention the legacy prefix forms in prose.
#
# This search used to use ripgrep, which `ubuntu-latest` does not ship.
# `rg ... || true` turned the resulting "command not found" into an empty result
# set, so the guard reported OK on every commit without ever reading a file.
# `grep` is in the runner's base image, and the self-test plus the exit-status
# check below make a search that cannot run fail instead of pass.
#
# PCRE2 negative-lookahead is avoided on purpose: some regex engines tickle
# catastrophic backtracking on `.*"..."` patterns over the whole repo.
#
# Run this after touching anything that names a reserved symbol.
set -euo pipefail

export LC_ALL=C

cd "$(dirname "$0")/.."
GIT_ROOT="$(git rev-parse --show-toplevel)"
cd "${GIT_ROOT}"

if ! command -v grep >/dev/null 2>&1; then
    echo "error: grep is required to verify reserved-prefix usage" >&2
    echo "       refusing to report success from a check that cannot run" >&2
    exit 1
fi

pattern='"cuda_oxide_(kernel|device|instantiate|const|artifact_anchor|ptx_merge_required|codegen_v1)_'

# Self-test.  The failure mode this guard has to survive is "silently stops
# matching anything", so require it to prove it can still match a known
# violation before a clean result is believed.
canary="$(mktemp -d)"
trap 'rm -rf "${canary}"' EXIT
mkdir -p "${canary}/crates"
printf 'const CANARY: &str = %s;\n' '"cuda_oxide_kernel_246e25db_canary"' \
    >"${canary}/crates/canary.rs"
if ! grep -rEn --include='*.rs' "${pattern}" "${canary}/crates" >/dev/null; then
    echo "error: reserved-prefix guard self-test failed: the search did not" >&2
    echo "       match a known violation, so a clean result means nothing" >&2
    exit 1
fi

# Host crates are crates.io / cutile-rs, not in-tree Oxide members. Search the
# nested SIMT tree only — the same scope as `crates/` on a flat Oxide checkout.
SEARCH_ROOTS=(cuda-oxide/crates)
for root in "${SEARCH_ROOTS[@]}"; do
    if [ ! -d "${root}" ]; then
        echo "error: reserved-prefix search root '${root}' does not exist" >&2
        exit 1
    fi
done

# grep exits 0 with matches, 1 with none, and >=2 on error.  Only 1 means
# clean; anything higher must fail rather than be mistaken for an empty result.
set +e
matches="$(grep -rEn --include='*.rs' "${pattern}" "${SEARCH_ROOTS[@]}" --exclude-dir=target)"
status=$?
set -e
if [ "${status}" -gt 1 ]; then
    echo "error: reserved-prefix search failed (grep exit ${status})" >&2
    exit 1
fi

violations="$(printf '%s\n' "${matches}" |
    grep -v '^cuda-oxide/crates/reserved-oxide-symbols/' |
    grep -vE ':[0-9]+:[[:space:]]*//' || true)"

if [ -n "${violations}" ]; then
    echo "error: hardcoded cuda_oxide_* prefix literals found outside reserved-oxide-symbols:" >&2
    printf '%s\n' "${violations}" | sed 's/^/  /' >&2
    echo >&2
    echo "Use the constants, builders, and predicates from" >&2
    echo "cuda-oxide/crates/reserved-oxide-symbols/ instead. See its README for the" >&2
    echo "layered API." >&2
    exit 1
fi

echo "OK: no hardcoded reserved-prefix literals outside reserved-oxide-symbols."
