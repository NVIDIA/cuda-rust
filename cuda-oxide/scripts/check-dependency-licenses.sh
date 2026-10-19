#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Verify dependency-licenses.csv still records every crate the workspace
# declares: each root-workspace member, and each directly declared third-party
# dependency (normal, dev, or build).  Run this after adding or removing a
# dependency or a workspace member.
#
# Scope, and why it stops where it does:
#
#   * Presence only, never versions.  The CSV records a snapshot while
#     Cargo.lock moves on its own, so comparing versions would fail on every
#     routine bump while saying nothing about licensing.
#   * Direct dependencies only, not the whole resolved graph.  cargo-deny
#     already enforces the license *policy* over every transitive crate
#     (deny.toml `[licenses]`).  This guard covers the other half: that the
#     human-readable inventory does not silently fall behind what the
#     workspace declares.  Transitive rows in the CSV are welcome, just not
#     required.
set -euo pipefail

# Pin the collation locale for every sort/comm in this script.  Both comm
# inputs are produced with byte-wise C ordering; without this, an ambient
# UTF-8 locale (e.g. en_US.UTF-8) makes GNU comm re-check the order under
# dictionary collation, reject the pair rustc-hash/rustc_apfloat with
# "input is not in sorted order", and abort the run via set -e.
export LC_ALL=C

cd "$(dirname "$0")/.."
GIT_ROOT="$(git rev-parse --show-toplevel)"
cd "${GIT_ROOT}"

CSV=cuda-oxide/dependency-licenses.csv

require_tool() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "error: $1 is required to verify ${CSV}" >&2
        echo "       refusing to report success from a check that cannot run" >&2
        exit 1
    fi
}
require_tool cargo
require_tool python3
require_tool git

test -s "${CSV}"

# Column 1 is the package name: always a bare crate name, never quoted and
# never containing a comma, so a plain field split is safe.  Later columns do
# use quoting (descriptions contain commas), which is why only column 1 is
# read this way.
#
# No CR handling is needed on this path, and the comment here used to claim
# otherwise. Two reasons: the committed file is LF-only since 8eb0dcf2 (#917)
# normalized it (111 CR bytes -> 0) while touching it for something else; and
# even on a CRLF checkout the CR sits before the newline, which puts it in the
# *last* field, never in the `cut -d, -f1` we read. Verified both ways -- CSV
# converted to CRLF, with and without a `tr -d '\r'` here, all four
# combinations agree.
recorded="$(tail -n +2 "${CSV}" | cut -d, -f1 | LC_ALL=C sort -u)"

# Self-test.  The failure mode a guard like this has to survive is "quietly
# stops seeing anything", so prove the CSV still parses into a plausible set
# of names before believing a clean result.
recorded_count="$(printf '%s\n' "${recorded}" | grep -c . || true)"
data_rows=$(($(wc -l <"${CSV}") - 1))
: "${recorded_count:=0}"
if [[ ${recorded_count} -lt 20 || ${data_rows} -lt 20 ]]; then
    echo "error: ${CSV} parse self-test failed: read ${recorded_count} package" \
        "names from ${data_rows} data rows" >&2
    echo "       the file layout changed; fix this script before trusting it" >&2
    exit 1
fi

# Members and directly declared third-party dependencies of one workspace.
#
# `--locked` so the guard reads the committed resolution rather than silently
# updating Cargo.lock to satisfy itself: a check that can rewrite its own input
# is not a check.
declared_crates() {
    cargo metadata --locked --format-version 1 --manifest-path "$1" | python3 -c '
import json, sys

metadata = json.load(sys.stdin)
workspace = set(metadata["workspace_members"])
members = [p for p in metadata["packages"] if p["id"] in workspace]
member_names = {p["name"] for p in members}

names = set(member_names)
for package in members:
    for dependency in package["dependencies"]:
        # Sibling path dependencies are already covered as workspace members.
        if dependency["name"] not in member_names:
            names.add(dependency["name"])

print("\n".join(sorted(names)))
'
}

# Every first-party workspace root, not just the root one.  `cargo metadata`
# stops at a `[workspace]` boundary, so one pass per root is the only way to
# reach them all.  The list is no longer a claim: the self-check below compares
# it against every tracked `[workspace]` manifest outside `examples/`, which the
# second half covers. Today that is four, and how each got here matters:
#
#   1. Cargo.toml -- the stable shared-host workspace.
#   2. cuda-oxide/Cargo.toml -- the nightly SIMT workspace.
#   3. cuda-oxide/crates/rustc-codegen-cuda/Cargo.toml -- its own `[workspace]` for the
#      rustc-private dylibs, so `-p` from the root cannot reach it and the root
#      `cargo metadata` stops at that boundary.  That is how the backend crate
#      itself, the largest first-party crate in the tree, sat unrecorded while
#      every other member had a row.  This is the second pass asked for in the
#      #662 review.
#   4. cuda-oxide/crates/cuda-macros/tests/device-only/Cargo.toml -- the fixture for
#      scripts/check-device-only-build.sh.  It declares `[workspace]` on
#      purpose: the point is a graph that does *not* contain `cuda-host`
#      (#701/#702), so it must resolve independently rather than share the root
#      lock.  That same boundary put it outside both passes above, and its
#      member `device-only-kernels` had no row.  #1043 closed the identical gap
#      for `cargo deny check` -- which judges the license *policy* -- and this
#      guard covers the other half, that the human-readable inventory does not
#      fall behind what a workspace declares.  Both halves need all three roots.
FIRST_PARTY_WORKSPACE_ROOTS=(
    Cargo.toml
    cuda-oxide/Cargo.toml
    cuda-oxide/crates/rustc-codegen-cuda/Cargo.toml
    cuda-oxide/crates/cuda-macros/tests/device-only/Cargo.toml
)

# The vendored rustlantis subtree also declares `[workspace]`. It is
# third-party code attributed in THIRD_PARTY_NOTICES, its dependencies are not
# ours to record, and check-spdx-headers.sh excludes it for the same reason.
VENDORED_WORKSPACE_ROOTS=(
    cuda-oxide/crates/fuzzer/rustlantis/Cargo.toml
)

# Ask Cargo which workspace owns every tracked manifest. This handles valid
# spellings such as `[ workspace ]` without raising the Python requirement.
# The result feeds both checks below:
#
#   Cargo.toml -> Cargo workspace root
#                      |-- named first-party or vendored root
#                      `-- example root with a tracked Cargo.lock
EXAMPLES_ROOT=cuda-oxide/crates/rustc-codegen-cuda/examples
workspace_roots() {
    git ls-files -z -- '*Cargo.toml' |
        while IFS= read -r -d '' manifest; do
            manifest_dir="$(dirname "${manifest}")"
            manifest_name="$(basename "${manifest}")"
            # rustup selects a toolchain from the working directory, not from
            # --manifest-path. Run beside each manifest so CUDA Oxide examples
            # with nightly-only Cargo features use the component pin.
            workspace_root="$(
                cd "${manifest_dir}"
                cargo locate-project --workspace --message-format plain --frozen \
                    --manifest-path "${manifest_name}"
            )" || {
                echo "error: Cargo could not locate the workspace for ${manifest}" >&2
                exit 1
            }
            case "${workspace_root}" in
                "${PWD}/"*) printf '%s\n' "${workspace_root#"${PWD}/"}" ;;
                *)
                    echo "error: Cargo returned a workspace root outside this checkout:" >&2
                    echo "       ${workspace_root}" >&2
                    exit 1
                    ;;
            esac
        done | LC_ALL=C sort -u
}
all_workspace_roots="$(workspace_roots)"
if [[ -z "${all_workspace_roots}" ]]; then
    echo "error: Cargo found no workspace roots; the scan broke" >&2
    exit 1
fi

# Nested cutile-rs is a separate product (own deny.toml / notices). Oxide's
# CSV does not inventory it yet, and .github/workflows/cargo-deny.yml leaves
# its roots out of the policy half for the same reason. When that changes,
# drop the CUTILE_ROOT filter below and add each cutile-rs `[workspace]` root
# to FIRST_PARTY_WORKSPACE_ROOTS (or VENDORED_WORKSPACE_ROOTS if it should
# stay out of the CSV).
CUTILE_ROOT=cutile-rs
on_disk_roots="$(
    printf '%s\n' "${all_workspace_roots}" |
        grep -v "^${EXAMPLES_ROOT}/" |
        grep -v "^${CUTILE_ROOT}/" |
        LC_ALL=C sort
)"
named_roots="$(
    printf '%s\n' "${FIRST_PARTY_WORKSPACE_ROOTS[@]}" "${VENDORED_WORKSPACE_ROOTS[@]}" |
        LC_ALL=C sort
)"
if [[ -z "${on_disk_roots}" ]]; then
    echo "error: found no tracked [workspace] manifests outside the examples" >&2
    echo "       tree; the scan broke, so a clean result means nothing" >&2
    exit 1
fi
unnamed="$(comm -23 <(printf '%s\n' "${on_disk_roots}") <(printf '%s\n' "${named_roots}"))"
if [[ -n "${unnamed}" ]]; then
    echo "error: these manifests declare [workspace] but no pass below reads them," >&2
    echo "       so nothing checks that ${CSV} records what they declare:" >&2
    printf '%s\n' "${unnamed}" | sed 's/^/  /' >&2
    echo >&2
    echo "Add each to FIRST_PARTY_WORKSPACE_ROOTS in $0, or to" >&2
    echo "VENDORED_WORKSPACE_ROOTS with the reason it is out of scope." >&2
    exit 1
fi
stale="$(comm -13 <(printf '%s\n' "${on_disk_roots}") <(printf '%s\n' "${named_roots}"))"
if [[ -n "${stale}" ]]; then
    echo "error: these names are listed as workspace roots in $0 but are no longer" >&2
    echo "       tracked manifests declaring [workspace]:" >&2
    printf '%s\n' "${stale}" | sed 's/^/  /' >&2
    echo >&2
    echo "A renamed or removed root left behind here is a permanent hole; drop it." >&2
    exit 1
fi

# Every example workspace is inventoried from its committed lock file. Compare
# the two sets before reading any package names so a new nested workspace
# cannot disappear merely because it has not committed a lock yet.
mapfile -d '' -t EXAMPLE_LOCKFILES < <(
    git ls-files -z -- "${EXAMPLES_ROOT}/**/Cargo.lock"
)
if [[ ${#EXAMPLE_LOCKFILES[@]} -lt 20 ]]; then
    echo "error: found only ${#EXAMPLE_LOCKFILES[@]} tracked example lock files;" >&2
    echo "       the scan broke, so a clean result means nothing" >&2
    exit 1
fi
example_roots="$(
    printf '%s\n' "${all_workspace_roots}" |
        grep "^${EXAMPLES_ROOT}/" |
        LC_ALL=C sort
)"
example_lock_roots="$(
    for lock in "${EXAMPLE_LOCKFILES[@]}"; do
        printf '%s/Cargo.toml\n' "${lock%/Cargo.lock}"
    done | LC_ALL=C sort -u
)"
missing_example_locks="$(
    comm -23 <(printf '%s\n' "${example_roots}") <(printf '%s\n' "${example_lock_roots}")
)"
if [[ -n "${missing_example_locks}" ]]; then
    echo "error: these example workspaces have no adjacent tracked Cargo.lock:" >&2
    printf '%s\n' "${missing_example_locks}" | sed 's/^/  /' >&2
    echo "Commit each lock before trusting the example dependency inventory." >&2
    exit 1
fi
orphan_example_locks="$(
    comm -13 <(printf '%s\n' "${example_roots}") <(printf '%s\n' "${example_lock_roots}")
)"
if [[ -n "${orphan_example_locks}" ]]; then
    echo "error: these tracked example locks have no Cargo workspace root:" >&2
    printf '%s\n' "${orphan_example_locks}" | sed 's/^/  /' >&2
    echo "Remove each stale lock or restore its workspace manifest." >&2
    exit 1
fi

required="$(
    for manifest in "${FIRST_PARTY_WORKSPACE_ROOTS[@]}"; do
        declared_crates "${manifest}"
    done | LC_ALL=C sort -u
)"

missing="$(comm -23 <(printf '%s\n' "${required}") <(printf '%s\n' "${recorded}"))"

if [[ -n "${missing}" ]]; then
    echo "error: ${CSV} is missing a row for:" >&2
    printf '%s\n' "${missing}" | sed 's/^/  /' >&2
    echo >&2
    echo "Every workspace member and every directly declared third-party" >&2
    echo "dependency needs a row.  See CONTRIBUTING.md ('If adding a new" >&2
    echo "dependency, update dependency-licenses.csv accordingly') and copy" >&2
    echo "the column layout from an existing row of the same kind." >&2
    exit 1
fi

echo "OK: ${CSV} records all $(printf '%s\n' "${required}" | grep -c .) declared crates."

# ---------------------------------------------------------------------------
# Second half: the example workspaces.
#
# Every example under cuda-oxide/crates/rustc-codegen-cuda/examples/ sets its own
# [workspace], so neither `cargo deny check` nor the check above resolves any
# of them -- both stop at the SIMT workspace boundary. Most examples declare
# only path dependencies on first-party crates and so bring nothing new, but a
# few link third-party code (tokio, rayon, libm, the shared cutile-rs crates),
# and that code is compiled by `cargo oxide run <example>` and by
# scripts/smoketest.sh without any license gate seeing it.
#
# Presence only, as above.  Lock files are parsed directly rather than through
# `cargo metadata`: resolving every example workspace separately would be slow
# and would need the network, and the lock files already record the resolved
# graph.  The search is recursive so a lockfile in a nested sub-workspace
# (e.g. cutile_inter_kernel/simt) is inventoried under its top-level example
# instead of escaping the guard.
#
# A package counts as covered when it is in the root graph, has a CSV row, or
# carries no `source` field.  That last case is a path dependency, which is
# first-party by construction.  Name matching is deliberately avoided -- some
# first-party crates are pulled from crates.io rather than by path (cuda-core
# and friends from cutile-rs), so a heuristic over names would misfile them.
# Examples whose third-party dependencies are deliberately out of inventory
# scope.  cutile_inter_kernel pulls the cutile compiler stack from cutile-rs,
# which resolves a further ~60 crates (wasm-bindgen, wit-bindgen, wasmparser,
# windows-targets) that exist in this tree only to build one interop example.
# Those crates still get a `cargo deny check` from
# check-example-license-policy.sh (the example is no longer exempt there);
# only the CSV rows are withheld.  Delete the entry to require the rows.
#
# Every name here is checked against the examples on disk below, so a typo or a
# rename fails the run instead of quietly exempting nothing -- or everything.
INVENTORY_EXEMPT_EXAMPLES=(cutile_inter_kernel)

examples_missing="$(python3 -c '
import os, re, sys

def packages(path):
    """(name, has_source) for every [[package]] in a Cargo.lock."""
    out = []
    for block in open(path).read().split("[[package]]")[1:]:
        name = re.search(r"^name = \"([^\"]+)\"", block, re.M)
        if name:
            out.append((name.group(1), re.search(r"^source = ", block, re.M) is not None))
    return out

covered = set()
for lock in (
    "Cargo.lock",
    "cuda-oxide/Cargo.lock",
    "cuda-oxide/crates/rustc-codegen-cuda/Cargo.lock",
):
    covered |= {name for name, _ in packages(lock)}

with open("cuda-oxide/dependency-licenses.csv", newline="") as handle:
    next(handle, None)
    covered |= {line.split(",")[0].strip().strip("\r") for line in handle if line.strip()}

separator = sys.argv.index("--")
exempt = set(sys.argv[1:separator])
locks = sorted(sys.argv[separator + 1:])
if len(locks) < 20:
    sys.exit("parse self-test failed: found %d example lock files" % len(locks))

examples_root = "cuda-oxide/crates/rustc-codegen-cuda/examples"

def example_of(lock):
    """Top-level example directory a lockfile belongs to, however deep it sits."""
    return os.path.relpath(lock, examples_root).split(os.sep)[0]

present = {example_of(lock) for lock in locks}
unknown = sorted(exempt - present)
if unknown:
    sys.exit("INVENTORY_EXEMPT_EXAMPLES names no such example: " + ", ".join(unknown))

seen = 0
findings = []
for lock in locks:
    example = example_of(lock)
    entries = packages(lock)
    seen += len(entries)
    if example in exempt:
        continue
    extra = sorted({n for n, sourced in entries if sourced and n not in covered})
    if extra:
        findings.append((example, extra))

if seen < 100:
    sys.exit("parse self-test failed: read %d packages from %d lock files" % (seen, len(locks)))

for example, extra in findings:
    print("%s: %s" % (example, " ".join(extra)))
' "${INVENTORY_EXEMPT_EXAMPLES[@]}" -- "${EXAMPLE_LOCKFILES[@]}")"

if [[ -n "${examples_missing}" ]]; then
    echo "error: ${CSV} is missing rows for third-party crates that example" >&2
    echo "       workspaces compile (neither cargo-deny nor the check above" >&2
    echo "       resolves these -- both stop at the SIMT workspace):" >&2
    printf '%s\n' "${examples_missing}" | sed 's/^/  /' >&2
    exit 1
fi

echo "OK: ${CSV} also covers every third-party crate the example workspaces pull."
