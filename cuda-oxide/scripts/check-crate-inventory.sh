#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Verify both crate inventories still list every workspace member: README.md's
# Crate Overview and the book's Crate Map.
#
# The overview is the only map of the tree a newcomer gets, and a crate absent
# from it is invisible: nothing else in the repo enumerates the members for a
# human, and no build fails when one goes unlisted.  `dialect-iket` and
# `iket-lower` were both missing when this guard was written -- 1,700 lines of
# compiler across two crates, with no row between them.
#
# Direction matters.  Every member needs a row; extra rows are fine and
# expected.  `rustc-codegen-cuda` is deliberately not a workspace member (it
# needs its own [workspace] for the rustc_private dylibs) and is listed anyway,
# which is right -- readers care about the crate, not about which workspace
# resolves it.  So this checks for members with no row, never for rows with no
# member.
#
# Members are read from the root Cargo.toml rather than from `cargo metadata`:
# the question is which crates the workspace declares, which is exactly what
# that list says, and reading it needs no cargo, no lockfile and no network.
#
# The book's Crate Map is the second inventory and it claims completeness in
# so many words -- "cuda-oxide is split into focused crates. Here is every one
# and its role" -- while sitting outside this guard's reach.  It listed 15 of 28
# when #970 found it, missing `dialect-iket` and `iket-lower` among eleven
# others: the same two crates whose absence from README.md is why this script
# exists.  One guard, both tables, so a new member cannot be absent from either.
#
# Run this after adding or removing a workspace member.
set -euo pipefail

export LC_ALL=C

cd "$(dirname "$0")/.."

MANIFEST=../Cargo.toml
# The Oxide overview stays at the git root. This script runs from cuda-oxide/.
README=../README.md
BOOK_MAP=cuda-oxide-book/compiler/architecture-overview.md

if ! command -v python3 >/dev/null 2>&1; then
    echo "error: python3 is required to verify the crate inventory" >&2
    echo "       refusing to report success from a check that cannot run" >&2
    exit 1
fi

test -s "${MANIFEST}"
test -s "${README}"
test -s "${BOOK_MAP}"

python3 - "${MANIFEST}" "${README}" "${BOOK_MAP}" <<'PY'
import re
import sys

manifest_path, readme_path, book_map_path = sys.argv[1], sys.argv[2], sys.argv[3]


def read(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()


manifest = read(manifest_path)
readme = read(readme_path)
book_map = read(book_map_path)

members_block = re.search(r"^members\s*=\s*\[(.*?)\]", manifest, re.M | re.S)
if not members_block:
    sys.exit(
        f"parse self-test failed: no `members = [...]` list in {manifest_path}; "
        "fix this script before trusting it"
    )

paths = re.findall(r'"([^"]+)"', members_block.group(1))

# A glob entry would silently name crates this list cannot resolve, so the
# guard has to say it went blind rather than pass over them.
globbed = [path for path in paths if "*" in path]
if globbed:
    sys.exit(
        "unsupported glob in workspace members: "
        + " ".join(globbed)
        + "; this guard resolves explicit paths only, so extend it before adding globs"
    )

# The directory name is the crate name everywhere in this tree, and the README
# tables key on the crate name.
members = sorted({path.rstrip("/").rsplit("/", 1)[-1] for path in paths})

if len(members) < 20:
    sys.exit(
        f"parse self-test failed: read {len(members)} members from {manifest_path}"
    )

# Only a section that is an inventory counts.  A crate named in passing
# elsewhere -- a command line, the pipeline diagram, a prose aside -- is not an
# inventory row, and accepting one would let a crate stay unlisted forever.
def rows_in_section(text, path, heading, stop, label):
    """First-column backticked names of every table row under `heading`."""
    section = re.search(
        rf"^## {re.escape(heading)}$(.*?)^{stop}", text, re.M | re.S
    )
    if not section:
        sys.exit(
            f"parse self-test failed: no `## {heading}` section in {path}; "
            "it was renamed or removed, so fix this script"
        )
    found = set(re.findall(r"^\|\s*`([^`]+)`\s*\|", section.group(1), re.M))
    if len(found) < 20:
        sys.exit(
            f"parse self-test failed: read {len(found)} crate rows from the "
            f"{label} in {path}"
        )
    return found


inventories = [
    (
        readme_path,
        rows_in_section(readme, readme_path, "Crate Overview", "## ", "Crate Overview"),
        "Crate Overview",
        "Add it to the table that fits (User-Facing, Compiler, or Build\n"
        "Tooling) and copy the column layout from a neighbouring row.",
    ),
    (
        book_map_path,
        rows_in_section(
            book_map, book_map_path, "Crate Map", "### ", "Crate Map"
        ),
        "Crate Map",
        'That table says "Here is every one and its role", so a member absent\n'
        "from it makes the page wrong as well as short.  Copy the role text\n"
        "from README.md's Crate Overview so the two inventories agree.",
    ),
]

failed = False
for path, listed, label, remedy in inventories:
    missing = [member for member in members if member not in listed]
    if not missing:
        continue
    failed = True
    print(f"error: {path}'s {label} has no row for:", file=sys.stderr)
    for member in missing:
        print(f"  {member}", file=sys.stderr)
    print(file=sys.stderr)
    print(remedy, file=sys.stderr)
    print(file=sys.stderr)

if failed:
    sys.exit(1)

print(
    f"OK: all {len(members)} workspace members are listed in "
    f"{readme_path}'s Crate Overview and {book_map_path}'s Crate Map."
)
PY
