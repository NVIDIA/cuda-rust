#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run from the repo root. Requires jq; no CUDA tools or GPU needed.
# Reuse Cargo's target directory: profile/env changes must invalidate the
# build without cargo clean. Change the runtime env on the same binary.
set -euo pipefail
command -v jq >/dev/null

check_default() {
    local expected="$1" override="$2"
    shift 2
    local executable runtime_mode
    local build_env=(env -u CUDA_RUST_DEBUG)
    if [[ "$override" != unset ]]; then
        build_env+=("CUDA_RUST_DEBUG=$override")
    fi
    executable=$("${build_env[@]}" cargo test -p cutile \
        --test debug_info --no-run --message-format=json "$@" | \
        jq -r 'select(.reason == "compiler-artifact" and .executable != null) | .executable')
    [[ -x "$executable" ]]
    for runtime_mode in none line full invalid; do
        CUTILE_TEST_EXPECT_DEBUG_INFO="$expected" CUDA_RUST_DEBUG="$runtime_mode" \
            "$executable" --exact build_time_default
    done
    env -u CUDA_RUST_DEBUG CUTILE_TEST_EXPECT_DEBUG_INFO="$expected" \
        "$executable" --exact build_time_default
}

# Profile defaults, and independence from debug assertions/build dependencies.
check_default full unset
check_default none unset --config profile.dev.debug=false
check_default full unset --config profile.dev.debug-assertions=false
check_default full unset --config profile.dev.build-override.debug=false
check_default none unset --config 'profile.dev.package.cutile-compiler.debug=false'
check_default none unset --release
check_default full unset --release --config profile.release.debug=true

# Cargo exposes only a boolean DEBUG to build scripts. Pin this limitation
# rather than claiming line-tables-only automatically selects device lines.
check_default full unset --config 'profile.dev.debug="line-tables-only"'
check_default full unset --config 'profile.dev.debug="limited"'

# Custom profiles work without inferring settings from PROFILE's name.
check_default full unset --profile cutile-debug-test \
    --config 'profile.cutile-debug-test.inherits="release"' \
    --config profile.cutile-debug-test.debug=true

# Explicit env overrides replace either profile default in both directions.
for mode in none line full; do
    check_default "$mode" "$mode" --config profile.dev.debug=true
    check_default "$mode" "$mode" --config profile.dev.debug=false
done

# Invalid configuration must fail loudly, not silently disable debug info.
if CUDA_RUST_DEBUG=invalid cargo check -p cutile-compiler; then
    echo "ERROR: invalid CUDA_RUST_DEBUG was accepted" >&2
    exit 1
fi

# Restore and verify the normal development-profile default.
check_default full unset
