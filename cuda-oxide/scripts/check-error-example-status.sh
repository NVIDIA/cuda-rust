#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Verify every error* example is documented in STATUS.md and listed in
# the ERROR_EXAMPLES array in smoketest.sh.  Run this after adding or
# removing an error* example.
set -euo pipefail

cd "$(dirname "$0")/.."

fail=0

# `mapfile` is a bash 4 builtin and `grep -P` is a GNU extension; macOS ships
# neither (bash 3.2, BSD grep), so this guard aborted there before it checked
# anything. Collect with a read loop and extract with `sed` instead -- both
# portable, and the extracted values are unchanged.
collect() {
    COLLECTED=()
    local line
    while IFS= read -r line; do
        COLLECTED+=("$line")
    done
}

# Examples that exist on disk.
collect < <(
    find crates/rustc-codegen-cuda/examples -mindepth 1 -maxdepth 1 \
        -type d -name 'error*' -exec basename {} \; | sort
)
on_disk=("${COLLECTED[@]+"${COLLECTED[@]}"}")

# Examples listed in STATUS.md (backtick-quoted names in the table).
collect < <(
    sed -n 's/^|[[:space:]]*`\([^`]*\)`.*/\1/p' \
        crates/rustc-codegen-cuda/STATUS.md | sort
)
in_status=("${COLLECTED[@]+"${COLLECTED[@]}"}")

# Examples listed in ERROR_EXAMPLES in smoketest.sh.
collect < <(
    sed -n 's/.*ERROR_EXAMPLES=(\([^)]*\)).*/\1/p' scripts/smoketest.sh \
        | tr ' ' '\n' | grep -v '^$' | sort
)
in_smoketest=("${COLLECTED[@]+"${COLLECTED[@]}"}")

# Parse self-test, the way this guard's eight siblings open.
#
# An empty on-disk list makes loop 1 vacuous and lets loop 2 pass (the
# directories it re-checks exist regardless of what the broken `find`
# returned), so the guard exits 0 with its success message -- reporting that
# everything is classified while having classified nothing. The same goes for
# every list coming back empty at once, say after a tree restructure. An empty
# STATUS.md or ERROR_EXAMPLES read alone did fail before this self-test, but
# per-example and blaming the wrong side. Each extraction is one `find`
# pattern or one `sed` expression away from empty: rename the example prefix,
# reflow STATUS.md's table, or wrap `ERROR_EXAMPLES=(` across lines. `set -e`
# does not help, because the failure is a command substitution producing no
# output rather than a non-zero status.
#
# So require all three to be non-empty and fail loudly, naming the extraction to
# fix, rather than trusting a clean result from an empty read.
self_test_failed=0
check_nonempty() {
    local count="$1" what="$2" how="$3"
    if [[ "${count}" -eq 0 ]]; then
        echo "error: parse self-test failed: found no ${what}" >&2
        echo "       ${how}" >&2
        self_test_failed=1
    fi
}
check_nonempty "${#on_disk[@]}" "error* example directories" \
    "fix the find in this script, or the examples really are all gone"
check_nonempty "${#in_status[@]}" "names in STATUS.md" \
    "its table layout changed; fix the sed that reads the first column"
check_nonempty "${#in_smoketest[@]}" "names in smoketest.sh ERROR_EXAMPLES" \
    "the array moved or wrapped across lines; that sed needs one line"
if [[ ${self_test_failed} -ne 0 ]]; then
    exit 1
fi

contains() {
    local needle="$1"; shift
    printf '%s\n' "$@" | grep -qx "$needle"
}

for ex in "${on_disk[@]}"; do
    if ! contains "$ex" "${in_status[@]+"${in_status[@]}"}"; then
        echo "error: $ex is not in STATUS.md" >&2; fail=1
    fi
    if ! contains "$ex" "${in_smoketest[@]+"${in_smoketest[@]}"}"; then
        echo "error: $ex is not in ERROR_EXAMPLES in smoketest.sh" >&2; fail=1
    fi
done

for ex in "${in_status[@]}"; do
    if [[ ! -d "crates/rustc-codegen-cuda/examples/$ex" ]]; then
        echo "error: STATUS.md lists '$ex' but no such directory exists" >&2; fail=1
    fi
done

# No reverse check for the smoketest array here on purpose:
# check-example-smoketest-contract.sh already rejects a name in any
# `*_EXAMPLES` array that is not a real example directory, across all thirteen
# arrays rather than just this one. Repeating it here would give the same
# condition two owners that can drift apart.

[[ $fail -eq 0 ]] && echo "OK: all error* examples are documented and classified."
exit $fail
