#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Verify a crate that only compiles kernels still builds with no host surface.
#
# #702 put the generated host items -- `#[cuda_module]`'s `LoadedModule` loader
# and launchers, and the `CudaKernel` impl for a bare `#[kernel]` -- behind a
# default-on `host` feature, so that a device-only kernel crate can take
# `cuda-macros = { default-features = false }` and stop dragging in
# cuda-host -> cuda-core -> cuda-bindings -> `cuda.h`. That is the whole point of
# #701: such a crate should type-check on a machine with no CUDA toolkit at all.
#
# Nothing exercised it. No example takes `default-features = false`, and every
# CI lane that builds a kernel crate has the toolkit installed, so the
# configuration was reachable only by a user who tried it.
#
# The failure mode is also quiet in the bad direction. Proc-macro features unify
# per build graph, so if any crate in a graph enables `cuda-macros/host`, every
# crate expanding these macros gets the host-emitting expansion -- including one
# that asked for `default-features = false` and has no cuda-host to resolve the
# generated `cuda_host::` paths. cuda-macros' own manifest documents that trap.
# A regression here does not fail loudly in this repo; it fails in a downstream
# crate, as E0433 on a path the user never wrote.
#
# So assert the two things #701 actually asked for, against the fixture in
# crates/cuda-macros/tests/device-only:
#
#   1. cuda-host, cuda-core and cuda-bindings are absent from its resolved
#      dependency graph. This is the property that removes the `cuda.h`
#      requirement, and it is checked from `cargo metadata` rather than by
#      reading the manifest, so an indirect reacquisition counts too.
#   2. It type-checks with the CUDA environment variables unset, which is what
#      "buildable without a toolkit" means operationally.
#
# Both are source-and-cargo only: no GPU, no toolkit, no backend build.
#
# Run this after touching cuda-macros' feature gating or the emitted host items.
set -euo pipefail

export LC_ALL=C

cd "$(dirname "$0")/.."

FIXTURE=crates/cuda-macros/tests/device-only
MANIFEST="${FIXTURE}/Cargo.toml"

require_tool() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "error: $1 is required to verify the device-only build" >&2
        echo "       refusing to report success from a check that cannot run" >&2
        exit 1
    fi
}
require_tool cargo
require_tool python3

test -s "${MANIFEST}"
test -s "${FIXTURE}/src/lib.rs"

# Parse self-test on the fixture itself. Every assertion below is only as
# meaningful as the fixture's configuration, and a fixture that quietly grew a
# `host` feature back, or stopped expanding the macros, would pass everything
# while proving nothing.
if ! grep -Fq 'default-features = false' "${MANIFEST}"; then
    echo "error: ${MANIFEST} no longer takes cuda-macros with" \
        "default-features = false" >&2
    echo "       that is the configuration under test; restore it" >&2
    exit 1
fi
for macro_use in '#\[kernel\]' '#\[cuda_module\]'; do
    if ! grep -Eq "${macro_use}" "${FIXTURE}/src/lib.rs"; then
        echo "error: ${FIXTURE}/src/lib.rs no longer uses ${macro_use}" >&2
        echo "       both macro entry points emit host items; keep both" >&2
        exit 1
    fi
done

# The CUDA environment is unset for both cargo invocations, so a toolkit that
# happens to be installed cannot mask a reacquired cuda-bindings dependency.
no_cuda_env() {
    env -u CUDA_HOME -u CUDA_TOOLKIT_PATH -u CUDA_TOOLKIT_TARGET_DIR "$@"
}

# `--locked` on both cargo invocations below, for the reason
# check-dependency-licenses.sh gives for its own `cargo metadata`: a check that
# can rewrite its own input is not a check. The fixture's Cargo.lock is
# tracked, and without the flag this guard re-resolves it in passing --
# verified by deleting the cuda-device package block from the committed lock,
# after which the guard still printed "OK: device-only graph is host-free
# (8 packages)" and left the lock silently repaired. That matters more here
# than elsewhere: the assertion is about the *resolved* graph, so a resolution
# free to move is not the one the repository committed.
graph="$(no_cuda_env cargo metadata --locked --format-version 1 \
    --manifest-path "${MANIFEST}" 2>/dev/null)" || {
    echo "error: cargo metadata failed for ${MANIFEST}" >&2
    echo "       if it reports a stale lock file, commit the updated" >&2
    echo "       ${FIXTURE}/Cargo.lock rather than letting the guard" >&2
    echo "       re-resolve it" >&2
    exit 1
}

printf '%s' "${graph}" | python3 -c '
import json, sys

HOST_SIDE = ("cuda-host", "cuda-core", "cuda-bindings")

metadata = json.load(sys.stdin)
names = sorted(package["name"] for package in metadata["packages"])

# Self-test: the fixture plus cuda-device, cuda-macros, reserved-oxide-symbols
# and the syn/quote/proc-macro2/unicode-ident that cuda-macros needs. A graph
# this small cannot have resolved nothing, and a graph that suddenly reads as
# one package means the query stopped seeing dependencies.
if len(names) < 4:
    sys.exit(f"parse self-test failed: resolved {len(names)} packages: {names}")

present = [name for name in names if name in HOST_SIDE]
if present:
    sys.exit(
        "error: the device-only fixture resolves host-side crates: "
        + " ".join(present)
        + "\n       a device-only kernel crate must not need cuda.h; see #701"
    )

print(f"OK: device-only graph is host-free ({len(names)} packages).")
'

if ! no_cuda_env cargo check --locked --quiet --manifest-path "${MANIFEST}" 2>/dev/null; then
    echo "error: the device-only fixture does not type-check without a CUDA" \
        "toolkit" >&2
    echo "       re-run without --quiet for the diagnostic:" >&2
    echo "       cargo check --locked --manifest-path ${MANIFEST}" >&2
    exit 1
fi

echo "OK: ${FIXTURE} type-checks with the CUDA environment unset."
