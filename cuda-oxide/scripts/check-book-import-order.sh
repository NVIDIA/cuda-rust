#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Keep the book's Rust blocks on rustfmt's import order.
#
# A reader pastes a book block into a `cargo oxide new` project, which is
# edition 2024, and the first `cargo fmt` rewrites the `use` lines: crates
# sorted, uppercase names ahead of lowercase (`{DisjointSlice, kernel, thread}`,
# not `{kernel, thread, DisjointSlice}`). The pasted code is correct either
# way, so nothing fails; the book just disagrees with what the toolchain
# produces, and with itself once some pages are fixed and others are not.
# #1241 found 47 blocks in that state, spread over 18 pages, after the same
# drift had already been fixed in the scaffold templates (#1193).
#
# A markdown block is compiled by nothing and formatted by nothing, so this
# guard formats the one part of a block that rustfmt can always judge on its
# own: the top-level `use` lines.
#
#   ```rust                       what the guard sees
#   use cuda_device::{a, B};  --> use cuda_device::{a, B};     rustfmt --check
#   use cuda_host::x;         --> use cuda_host::x;            on just this
#                                                              region
#   #[kernel]                     (never looked at)
#   pub fn k(x: &mut f32,
#            y: u32) {...}        one-param-per-line teaching
#   ```                           layout stays the author's
#
# Scope, deliberately narrow so the guard stays precise:
#
#   * Unindented `use` items only, in every ```rust block under
#     cuda-oxide-book/. Each contiguous run of them is one rustfmt input, so a
#     blank line between groups keeps the groups apart, as it does in a crate.
#   * Nothing else in the block is formatted. Signature layout, comments and
#     bodies are teaching text and are the author's call. A partial snippet
#     with no enclosing item is fine too: its `use` lines parse on their own.
#   * Book only. Crate and example READMEs are not a paste-into-a-project
#     surface in the same way and stay out.
#
# rustfmt comes from rust-toolchain.toml's component list, through the rustup
# proxy, so the order checked here is the order the pinned nightly writes.
# `--edition 2024` matches the edition `cargo oxide new` scaffolds; under 2021
# rustfmt keeps the old case-insensitive order and this guard would say
# nothing. A `use` line rustfmt cannot parse is reported too: a reader pastes
# it, so it has to be a `use` line.
#
# Run this after editing any Rust block in the book; the diff it prints is the
# fix.
set -euo pipefail

export LC_ALL=C

cd "$(dirname "$0")/.."

BOOK=cuda-oxide-book

for tool in python3 rustfmt; do
    if ! command -v "${tool}" >/dev/null 2>&1; then
        echo "error: ${tool} is required to verify the book's import order" >&2
        echo "       refusing to report success from a check that cannot run" >&2
        exit 1
    fi
done

test -d "${BOOK}"

# One call up front, so a first-run toolchain install (rustup installs the pin
# on demand on a fresh CI runner) happens here and its progress output cannot
# be mistaken for a rustfmt diagnostic on a book block below.
RUSTFMT_VERSION="$(rustfmt --version)"

mapfile -t pages < <(git ls-files "${BOOK}/*.md" "${BOOK}/**/*.md" | sort -u)

python3 - "${RUSTFMT_VERSION}" "${pages[@]}" <<'PY'
import os
import re
import subprocess
import sys
import tempfile

rustfmt_version, pages = sys.argv[1], sys.argv[2:]

RUSTFMT = ["rustfmt", "--edition", "2024", "--check", "--color=never"]
FENCE_OPEN = re.compile(r"^```rust\b")
FENCE_CLOSE = re.compile(r"^```")
# Unindented `use`, with or without a visibility. Indented ones live inside a
# body or a `mod` and are the author's layout, not a top-level import region.
USE_ITEM = re.compile(r"^(pub(\([^)]*\))?\s+)?use\s")
ATTRIBUTE = re.compile(r"^#\[")

if len(pages) < 20:
    sys.exit(f"parse self-test failed: found {len(pages)} book pages")


def blocks(path):
    """(fence line, block lines) for every ```rust block in a page."""
    out, current, start = [], None, 0
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            text = line.rstrip("\n")
            if current is None:
                if FENCE_OPEN.match(text):
                    current, start = [], number
            elif FENCE_CLOSE.match(text):
                out.append((start, current))
                current = None
            else:
                current.append(text)
    return out


def import_runs(lines):
    """Contiguous runs of top-level `use` items, blank separators kept.

    Yields (first line index, last line index inclusive, text). A run ends at
    the first non-blank line that is not a top-level `use` item; blank lines
    inside a run are group separators, and trailing ones are dropped.
    """
    i, n = 0, len(lines)
    while i < n:
        # Attributes directly above a `use` belong to it.
        j = i
        while j < n and ATTRIBUTE.match(lines[j]):
            j += 1
        if j >= n or not USE_ITEM.match(lines[j]):
            i += 1
            continue
        start, end, k = i, None, i
        while k < n:
            if ATTRIBUTE.match(lines[k]):
                k += 1
                continue
            if USE_ITEM.match(lines[k]):
                depth = lines[k].count("{") - lines[k].count("}")
                while not (lines[k].rstrip().endswith(";") and depth <= 0):
                    k += 1
                    if k >= n:
                        break
                    depth += lines[k].count("{") - lines[k].count("}")
                end = min(k, n - 1)
                k += 1
                continue
            if lines[k].strip() == "":
                k += 1
                continue
            break
        yield start, end, "\n".join(lines[start : end + 1]) + "\n"
        i = end + 1


def rustfmt_check(text, workdir):
    path = os.path.join(workdir, "imports.rs")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    result = subprocess.run(RUSTFMT + [path], capture_output=True, text=True)
    if result.returncode == 0:
        return None
    if result.stdout.startswith("Diff in"):
        # Keep only the diff body; each hunk header names the temp file.
        return "\n".join(
            line
            for line in result.stdout.splitlines()
            if line.rstrip() and not line.startswith("Diff in")
        )
    return "rustfmt could not parse this import region:\n" + result.stderr.strip()


checked_blocks = 0
checked_runs = 0
failures = []
with tempfile.TemporaryDirectory() as workdir:
    for page in pages:
        for fence, lines in blocks(page):
            runs = list(import_runs(lines))
            if not runs:
                continue
            checked_blocks += 1
            reports = []
            for start, _, text in runs:
                checked_runs += 1
                report = rustfmt_check(text, workdir)
                if report is not None:
                    reports.append((fence + 1 + start, report))
            if reports:
                failures.append((page, fence, reports))

# Both halves have to have found something, or a rot in the fence or `use`
# regex reads as a clean book rather than a broken check.
if checked_blocks < 50:
    sys.exit(
        f"parse self-test failed: found {checked_blocks} Rust blocks with "
        "top-level imports in the book"
    )

if failures:
    print(
        "error: the book's Rust blocks disagree with rustfmt on import order:",
        file=sys.stderr,
    )
    for page, fence, reports in failures:
        for line, report in reports:
            print(f"\n  {page}:{line} (block opened at line {fence})", file=sys.stderr)
            for text in report.splitlines():
                print(f"    {text}", file=sys.stderr)
    print(file=sys.stderr)
    print(
        f"{len(failures)} block(s). Apply the diff above (only the `use` lines "
        "change); the order is what",
        file=sys.stderr,
    )
    print(
        f"`cargo fmt` writes in an edition-2024 project ({rustfmt_version}).",
        file=sys.stderr,
    )
    sys.exit(1)

print(
    f"OK: {checked_runs} import regions in {checked_blocks} Rust blocks across "
    f"{len(pages)} book pages match {rustfmt_version} (edition 2024)."
)
PY
