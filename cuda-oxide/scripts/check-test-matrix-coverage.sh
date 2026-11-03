#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Verify every workspace member is either tested by CI or declared untested.
#
# `unit-tests.yml`'s matrix decides which crates get `cargo test`.  A member
# absent from it is not reported anywhere: the workflow passes, `just check`
# passes, and the crate's tests simply never run.  #971 found four in that
# state holding 107 `#[test]` functions between them -- `dialect-ptx` (43),
# `ptx-parse` (35), `iket-lower` (22) and `dialect-iket` (7).  `dialect-iket`
# and `iket-lower` had been uncovered for months.
#
# The workflow already carries the intent, as a comment listing what is out of
# scope on purpose:
#
#     # Crates intentionally not in this matrix:
#     #   * `cuda-bindings` - generated sys bindings covered transitively.
#     #   * `fuzzer` - differential-testing infra with no unit tests.
#     #   * `rustc-codegen-cuda` - its own [workspace] ...
#
# That comment reading as complete is what made the four invisible.  This guard
# makes it complete: a member must appear in the matrix or be named in that
# list, and anything in neither fails the run.  Adding a crate to the exclusion
# list stays a one-line, reviewable act -- which is the point.  Removing the
# guesswork, not the choice.
#
# Also checked, because a stale list is as bad as a short one: every name in the
# exclusion comment must still be a real member (or `rustc-codegen-cuda`, which
# is deliberately not one), so a renamed or deleted crate cannot leave a
# permanent hole behind.
#
# The Justfile's `test` and `test-cuda` recipes are checked against the same
# matrix, since their own comment is the contract -- "Mirrors
# .github/workflows/unit-tests.yml; keep the two in step" -- and nothing
# enforced it.
#
# Reads three text files: no cargo, no network, no toolchain.
#
# Run this after adding a workspace member or editing the matrix.
set -euo pipefail

export LC_ALL=C

cd "$(dirname "$0")/.."

MANIFEST=Cargo.toml
WORKFLOW=../.github/workflows/unit-tests.yml
JUSTFILE=Justfile

if ! command -v python3 >/dev/null 2>&1; then
    echo "error: python3 is required to verify test-matrix coverage" >&2
    echo "       refusing to report success from a check that cannot run" >&2
    exit 1
fi

test -s "${MANIFEST}"
test -s "${WORKFLOW}"
test -s "${JUSTFILE}"

python3 - "${MANIFEST}" "${WORKFLOW}" "${JUSTFILE}" <<'PY'
import re
import sys

manifest_path, workflow_path, justfile_path = sys.argv[1:4]


def read(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()


manifest, workflow, justfile = (
    read(manifest_path),
    read(workflow_path),
    read(justfile_path),
)

# Workspace members, read from the manifest for the reason
# check-crate-inventory.sh gives: the question is what the workspace declares,
# and reading it needs no cargo and no lockfile.
members_block = re.search(r"^members\s*=\s*\[(.*?)\]", manifest, re.M | re.S)
if not members_block:
    sys.exit(
        f"parse self-test failed: no `members = [...]` list in {manifest_path}; "
        "fix this script before trusting it"
    )
paths = re.findall(r'"([^"]+)"', members_block.group(1))
globbed = [path for path in paths if "*" in path]
if globbed:
    sys.exit(
        "unsupported glob in workspace members: "
        + " ".join(globbed)
        + "; this guard resolves explicit paths only, so extend it before adding globs"
    )
members = sorted({path.rstrip("/").rsplit("/", 1)[-1] for path in paths})
if len(members) < 20:
    sys.exit(
        f"parse self-test failed: read {len(members)} members from {manifest_path}"
    )

# The matrix entries.  `- package: <name>` is the only shape the workflow uses.
matrix = set(re.findall(r"^\s+- package:\s*([a-z0-9-]+)\s*$", workflow, re.M))
if len(matrix) < 15:
    sys.exit(
        f"parse self-test failed: read {len(matrix)} matrix packages from "
        f"{workflow_path}; the entry shape changed, fix this script"
    )

# The exclusion comment.  Only backticked names on a `#   * ` bullet count, so
# prose in the surrounding paragraphs cannot silently exempt a crate.
excluded_block = re.search(
    r"Crates intentionally not in this matrix:(.*?)\n\s+include:", workflow, re.S
)
if not excluded_block:
    sys.exit(
        "parse self-test failed: no `Crates intentionally not in this matrix:` "
        f"comment found before `include:` in {workflow_path}; it moved or was "
        "reworded, so fix this script rather than trusting an empty exclusion set"
    )
excluded = set(re.findall(r"^\s*#\s+\*\s+`([a-z0-9-]+)`", excluded_block.group(1), re.M))
if not excluded:
    sys.exit(
        "parse self-test failed: the exclusion comment in "
        f"{workflow_path} yielded no names; the bullet shape changed"
    )

failures = []

# 1. Every member is tested or declared untested.
uncovered = [m for m in members if m not in matrix and m not in excluded]
if uncovered:
    failures.append(
        "these workspace members are neither in the unit-tests matrix nor named\n"
        "  in its exclusion comment, so their tests never run:\n    "
        + "\n    ".join(uncovered)
        + "\n  Add a `- package: <name>` entry to .github/workflows/unit-tests.yml,\n"
        "  or name the crate in the `Crates intentionally not in this matrix:`\n"
        "  comment with the reason."
    )

# 2. The exclusion list names real crates.  `rustc-codegen-cuda` is excluded and
#    is deliberately not a member (its own [workspace] for the rustc_private
#    dylibs), so it is the one accepted non-member.
known = set(members) | {"rustc-codegen-cuda"}
phantom = sorted(name for name in excluded if name not in known)
if phantom:
    failures.append(
        "the exclusion comment names crates that are not workspace members:\n    "
        + "\n    ".join(phantom)
        + "\n  A renamed or deleted crate left behind here is a permanent hole;\n"
        "  drop the bullet."
    )

# 3. A matrix entry must be a real member too, or the job runs `cargo test -p`
#    on a name cargo cannot resolve.
unknown = sorted(name for name in matrix if name not in members)
if unknown:
    failures.append(
        "the matrix names packages that are not workspace members:\n    "
        + "\n    ".join(unknown)
    )

# 4. The Justfile mirrors the matrix, which its own comment promises.
recipes = {}
for name in ("test", "test-cuda"):
    body = re.search(rf"^{name}:\n(.*?)(?=\n^[a-z@#]|\Z)", justfile, re.M | re.S)
    if not body:
        sys.exit(
            f"parse self-test failed: no `{name}:` recipe in {justfile_path}; "
            "it was renamed, so fix this script"
        )
    # `[a-z]` first, deliberately: `test-cuda` contains `ldconfig -p 2>/dev/null`,
    # and a looser `[a-z0-9-]+` reads that redirect as a package named "2".
    recipes[name] = set(re.findall(r"-p ([a-z][a-z0-9-]*)", body.group(1)))
just = recipes["test"] | recipes["test-cuda"]
if len(just) < 15:
    sys.exit(
        f"parse self-test failed: read {len(just)} packages from {justfile_path}'s "
        "test recipes; the `-p` spelling changed"
    )
only_ci = sorted(matrix - just)
only_just = sorted(just - matrix)
if only_ci or only_just:
    detail = []
    if only_ci:
        detail.append("in the matrix but not in `just test`/`test-cuda`: " + " ".join(only_ci))
    if only_just:
        detail.append("in `just test`/`test-cuda` but not in the matrix: " + " ".join(only_just))
    failures.append(
        "the Justfile no longer mirrors the matrix, which its own comment\n"
        '  promises ("Mirrors .github/workflows/unit-tests.yml; keep the two in\n'
        '  step"):\n    ' + "\n    ".join(detail)
    )

if failures:
    print("error: test-matrix coverage is incomplete", file=sys.stderr)
    for failure in failures:
        print(f"  {failure}", file=sys.stderr)
        print(file=sys.stderr)
    sys.exit(1)

print(
    f"OK: all {len(members)} workspace members are covered -- {len(matrix)} in the "
    f"unit-tests matrix, {len(excluded)} declared untested -- and the Justfile's "
    "test recipes mirror the matrix."
)
PY
