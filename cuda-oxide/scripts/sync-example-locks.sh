#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Keep every example workspace's Cargo.lock resolving the same dependency
# versions.
#
# Why this matters: each example under crates/rustc-codegen-cuda/examples/
# is its own cargo workspace with its own committed Cargo.lock. CI builds
# them all into one shared CARGO_TARGET_DIR, and cargo keys its build cache
# on the resolved versions of a unit's whole dependency subtree. When two
# examples pin different versions of, say, syn or memchr, the first example
# to use each resolution recompiles a private variant of the entire shared
# chain (bindgen re-run over cuda.h, cuda-bindings, cuda-core, cuda-device,
# cuda-host, ...). Before the locks were synced, 34 distinct resolutions
# existed across 219 examples and each one cost a 15-70s dependency-tree
# rebuild in the examples-compile CI job.
#
# Usage:
#   scripts/sync-example-locks.sh --check
#       Verify that all example locks agree (used as a CI gate; no network,
#       needs only python3). Two invariants:
#       1. For every crate name and semver-compatibility bucket (major
#          version, or 0.minor for 0.x crates), at most one exact version
#          may appear across all example locks.
#       2. No two example workspaces define local crates with the same
#          (package name, version). Cargo does not key workspace-member
#          cache slots on the manifest path, so under the shared
#          CARGO_TARGET_DIR two same-named local crates alias one slot and
#          the second example silently links whichever library compiled
#          first (found the hard way with two `kernel-lib v0.1.0` crates).
#
#   scripts/sync-example-locks.sh --bump
#       Re-resolve every example workspace with `cargo update` in one sweep
#       (network). Because all workspaces resolve against the same registry
#       snapshot, shared dependencies land on identical versions. Workspaces
#       whose lock contains git dependencies are skipped (a bare `cargo
#       update` would also move the git pin); align them afterwards.
#
#   scripts/sync-example-locks.sh --align [example-dir ...]
#       Align drifted locks to the majority version per crate using targeted
#       `cargo update -p <name>@<from> --precise <to>` (never touches git
#       pins). With no arguments, aligns every drifted lock. Use this for a
#       newly added example, or for git-dependency workspaces after --bump.
#
# Adding a new example? Copy an existing example's Cargo.lock (e.g.
# vecadd's) into the new workspace before the first build; cargo keeps the
# shared pins and adds only the new crates. Then run --check.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
EXAMPLES_ROOT="crates/rustc-codegen-cuda/examples"

usage() {
    # The doc block above: everything after the shebang + SPDX lines, up to
    # the first non-comment line. No hardcoded line range to fall stale.
    awk 'NR > 3 { if ($0 !~ /^#/) exit; sub(/^# ?/, ""); print }' "${BASH_SOURCE[0]}"
}

# Print every example Cargo.lock path (top-level workspaces and nested
# device workspaces alike), one per line.
all_locks() {
    find "${EXAMPLES_ROOT}" -name Cargo.lock | sort
}

# ---- check ---------------------------------------------------------------

# Exit 0 when every (crate name, semver bucket) resolves to one exact
# version across all example locks AND no two example workspaces define
# same-named local crates; otherwise list each conflict.
run_check() {
    LOCK_PATHS="$(all_locks)" EXAMPLES_ROOT="${EXAMPLES_ROOT}" python3 - <<'PY'
import os
import re
import sys
from collections import defaultdict

# Cargo's compatibility rule: leftmost non-zero component. Every 0.0.z
# version is its own compatibility range, so it gets its own bucket
# (two 0.0.z pins of one crate can legitimately coexist).
def bucket(version):
    parts = version.split(".")
    if parts[0] != "0":
        return parts[0]
    if len(parts) > 1 and parts[1] != "0":
        return "0." + parts[1]
    return version

# (name, bucket) -> (version, source) -> [lock paths]. The source string is
# part of the resolved identity: the same version from two git revs (or git
# vs registry) is two distinct cargo package IDs and two cache slots.
seen = defaultdict(lambda: defaultdict(list))
lock_paths = [line.strip() for line in os.environ["LOCK_PATHS"].splitlines() if line.strip()]
if len(lock_paths) < 100:
    sys.exit(f"parse self-test failed: only {len(lock_paths)} example locks found")

package_re = re.compile(
    r'^\[\[package\]\]\nname = "([^"]+)"\nversion = "([^"]+)"\nsource = "([^"]+)"', re.M
)
for path in lock_paths:
    text = open(path, encoding="utf-8").read()
    entries = package_re.findall(text)
    if not entries:
        sys.exit(f"parse self-test failed: no registry/git packages parsed in {path}")
    for name, version, source in entries:
        seen[(name, bucket(version))][(version, source)].append(path)

failed = False

conflicts = {key: vers for key, vers in seen.items() if len(vers) > 1}
if conflicts:
    failed = True
    print(f"error: {len(conflicts)} crate(s) resolve differently across example locks:\n")
    for (name, buck), versions in sorted(conflicts.items()):
        print(f"  {name} (compat range {buck}.*):")
        for (version, source), paths in sorted(versions.items()):
            sample = ", ".join(paths[:3]) + (" ..." if len(paths) > 3 else "")
            origin = source if source.startswith("git+") else "registry"
            print(f"    {version} ({origin})  in {len(paths)} lock(s): {sample}")
    print(
        "\nFix: for a new example, copy an existing example's Cargo.lock before"
        "\nthe first build; for drift, run scripts/sync-example-locks.sh --align"
        "\n(or --bump to refresh every pin in one sweep).\n"
    )

# Invariant 2: local crate (name, version) must be unique across example
# workspaces. Cargo's cache slots for workspace members are not keyed on the
# manifest path, so duplicates alias one slot in the shared target dir and
# the second example links whichever library compiled first.
name_re = re.compile(r'^name\s*=\s*"([^"]+)"', re.M)
version_re = re.compile(r'^version\s*=\s*"([^"]+)"', re.M)
local_crates = defaultdict(set)
manifest_count = 0
for dirpath, dirnames, filenames in os.walk(os.environ["EXAMPLES_ROOT"]):
    dirnames[:] = [d for d in dirnames if d not in ("target", ".oxide-artifacts", "src")]
    if "Cargo.toml" not in filenames:
        continue
    manifest_count += 1
    text = open(os.path.join(dirpath, "Cargo.toml"), encoding="utf-8").read()
    package = text.split("[package]", 1)
    if len(package) < 2:
        continue
    body = package[1].split("\n[", 1)[0]
    name, version = name_re.search(body), version_re.search(body)
    if name and version:
        local_crates[(name.group(1), version.group(1))].add(dirpath)
if manifest_count < 100:
    sys.exit(f"parse self-test failed: only {manifest_count} example manifests found")
duplicated = {key: dirs for key, dirs in local_crates.items() if len(dirs) > 1}
if duplicated:
    failed = True
    print("error: same (package name, version) defined in more than one example workspace:\n")
    for (name, version), dirs in sorted(duplicated.items()):
        print(f"  {name} v{version}:")
        for d in sorted(dirs):
            print(f"    {d}")
    print(
        "\nFix: rename one of the packages (or bump its version); same-named"
        "\nlocal crates alias one cache slot in CI's shared CARGO_TARGET_DIR."
    )

if failed:
    sys.exit(1)
print(
    f"OK: {len(lock_paths)} example locks agree on one version per crate "
    f"({len(seen)} crate/bucket pairs), and {len(local_crates)} local "
    f"crate identities are unique."
)
PY
}

# ---- bump ----------------------------------------------------------------

run_bump() {
    local lock dir skipped=0 updated=0
    while IFS= read -r lock; do
        dir="$(dirname "${lock}")"
        if grep -q '^source = "git+' "${lock}"; then
            echo "skip (git pins): ${dir}  -- align it afterwards"
            skipped=$((skipped + 1))
            continue
        fi
        echo "cargo update: ${dir}"
        cargo update --manifest-path "${dir}/Cargo.toml" --quiet
        updated=$((updated + 1))
    done < <(all_locks)
    echo "updated ${updated} lock(s), skipped ${skipped} git-pinned lock(s)"
}

# ---- align ---------------------------------------------------------------

# Emit `<manifest-path>\t<name>@<from>\t<to>` lines for every package whose
# exact version disagrees with the majority across all locks, restricted to
# the requested lock dirs (or all locks when none are given).
plan_align() {
    LOCK_PATHS="$(all_locks)" EXAMPLES_ROOT="${EXAMPLES_ROOT}" python3 - "$@" <<'PY'
import os
import re
import sys
from collections import defaultdict

examples_root = os.environ["EXAMPLES_ROOT"]

# A target selects one example workspace by its directory name (nested
# device workspaces included). Exact path-component matching: "gemm" must
# not also select "tiled_gemm".
def normalize(target):
    target = target.rstrip("/")
    for prefix in (examples_root + "/", "./" + examples_root + "/"):
        if target.startswith(prefix):
            target = target[len(prefix):]
    return target

targets = {normalize(t) for t in sys.argv[1:]}

def selected(lock_dir):
    if not targets:
        return True
    rel = os.path.relpath(lock_dir, examples_root)
    return any(rel == t or rel.startswith(t + "/") for t in targets)

# Cargo's compatibility rule: leftmost non-zero component; each 0.0.z
# version is its own range. Keep textually identical to run_check's copy.
def bucket(version):
    parts = version.split(".")
    if parts[0] != "0":
        return parts[0]
    if len(parts) > 1 and parts[1] != "0":
        return "0." + parts[1]
    return version

# Semver-aware ordering for majority tie-breaks: numeric components, with
# a release preferred over any pre-release of the same core.
def ver_key(version):
    core, _, pre = version.partition("-")
    numeric = tuple(int(p) if p.isdigit() else 0 for p in core.split("."))
    return (numeric, pre == "", pre)

package_re = re.compile(
    r'^\[\[package\]\]\nname = "([^"]+)"\nversion = "([^"]+)"\nsource = "([^"]+)"', re.M
)
lock_paths = [line.strip() for line in os.environ["LOCK_PATHS"].splitlines() if line.strip()]
per_lock = {}
seen = defaultdict(lambda: defaultdict(list))
for path in lock_paths:
    entries = package_re.findall(open(path, encoding="utf-8").read())
    per_lock[path] = entries
    for name, version, source in entries:
        seen[(name, bucket(version))][(version, source)].append(path)

majority = {
    key: max(
        versions.items(),
        key=lambda item: (len(item[1]), ver_key(item[0][0]), item[0]),
    )[0]
    for key, versions in seen.items()
}

for path, entries in sorted(per_lock.items()):
    lock_dir = os.path.dirname(path)
    if not selected(lock_dir):
        continue
    manifest = os.path.join(lock_dir, "Cargo.toml")
    for name, version, source in entries:
        want_version, want_source = majority[(name, bucket(version))]
        if (version, source) == (want_version, want_source):
            continue
        # `cargo update -p X@ver --precise <to>` on a git dependency treats
        # <to> as a git revspec, which either errors confusingly or silently
        # moves the pin. Git-side drift is reported, never auto-planned.
        if source.startswith("git+") or want_source.startswith("git+"):
            print(
                f"note: {manifest}: {name}@{version} ({source}) disagrees with "
                f"the majority {want_version} ({want_source}); git pins are "
                f"never auto-aligned -- update the git reference by hand",
                file=sys.stderr,
            )
            continue
        if want_version != version:
            print(f"{manifest}\t{name}@{version}\t{want_version}")
        else:
            print(
                f"note: {manifest}: {name}@{version} resolves from {source} "
                f"instead of {want_source}; fix the registry source by hand",
                file=sys.stderr,
            )
PY
}

run_align() {
    local plan manifest spec to pass count=0 output
    # Updating one pin can cascade to version-locked siblings (the futures
    # family moves as a unit), which invalidates later entries of the same
    # plan. Apply, tolerate exactly the "no longer matches" staleness, and
    # re-plan until a pass finds nothing left.
    for pass in 1 2 3 4 5 6 7 8 9 10; do
        plan="$(plan_align "$@")"
        if [[ -z "${plan}" ]]; then
            if [[ ${count} -eq 0 && ${pass} -eq 1 ]]; then
                echo "nothing to align"
            else
                echo "aligned ${count} package pin(s)"
            fi
            return 0
        fi
        while IFS=$'\t' read -r manifest spec to; do
            echo "align: ${manifest}: ${spec} -> ${to}"
            if ! output="$(cargo update --manifest-path "${manifest}" \
                -p "${spec}" --precise "${to}" --quiet 2>&1)"; then
                if grep -q "did not match any packages" <<<"${output}"; then
                    echo "  (already moved by an earlier update; will re-plan)"
                    continue
                fi
                echo "${output}" >&2
                return 1
            fi
            count=$((count + 1))
        done <<<"${plan}"
    done
    echo "error: alignment did not converge after 10 passes" >&2
    return 1
}

# ---- entry ---------------------------------------------------------------

case "${1:---check}" in
    --check) run_check ;;
    --bump)  run_bump ;;
    --align) shift; run_align "$@" ;;
    -h|--help) usage ;;
    *) echo "error: unknown mode '$1'" >&2; usage >&2; exit 2 ;;
esac
