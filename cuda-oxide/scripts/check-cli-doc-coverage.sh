#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Verify the book's Command reference lists every `cargo oxide` subcommand.
#
# The table in `cuda-oxide-book/getting-started/installation.md` is the only
# complete list of the CLI surface anywhere in the docs, and nothing fails when
# it falls behind: adding a subcommand to the clap enum is a one-line change
# that leaves the table silently short. Four commands (`test`, `fmt`, `update`,
# `emit-ltoir`) were already missing when that table was written, so the drift
# is not hypothetical.
#
# Both directions are checked. A missing row is the common case; a row for a
# command that no longer exists is the other one, and it is worse, because a
# reader trusts it and gets `unrecognized subcommand`.
#
# Subcommands are read from the `Commands` enum in
# `crates/cargo-oxide/src/main.rs` rather than from `cargo oxide --help`. The
# enum is the definition, reading it needs no cargo, no toolchain and no build,
# and it keeps this guard in the same source-only class as its siblings -- a
# guard that needs a 15-minute backend build to answer a documentation question
# would not be run.
#
# Two kinds of command are deliberately not required to appear:
#
#   * anything carrying `hide = true`, which is hidden from `--help` itself
#     (`__materializer-provenance` today, internal plumbing);
#   * clap's generated `help`, which is not in the enum at all, so reading the
#     enum excludes it without a special case.
#
# Descriptions are out of scope. clap's short help is a variant's doc-comment
# first line, while the table's cells deliberately add MyST backticks around
# names like `cuda-gdb` and `setup`; comparing them literally would fail on
# correct prose, and normalising invites the brittleness this guard exists to
# avoid. What matters is that every command is listed and no listed command is
# fictional.
#
# Run this after adding, renaming or removing a subcommand.
set -euo pipefail

export LC_ALL=C

cd "$(dirname "$0")/.."

MAIN=crates/cargo-oxide/src/main.rs
BOOK=cuda-oxide-book/getting-started/installation.md

if ! command -v python3 >/dev/null 2>&1; then
    echo "error: python3 is required to verify the CLI command reference" >&2
    echo "       refusing to report success from a check that cannot run" >&2
    exit 1
fi

test -s "${MAIN}"
test -s "${BOOK}"

python3 - "${MAIN}" "${BOOK}" <<'PY'
import re
import sys

main_path, book_path = sys.argv[1], sys.argv[2]


def read(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()


main_src = read(main_path)
book_src = read(book_path)

# The enum body, from its header to the first column-0 `}`. Variant bodies are
# indented deeper, so they cannot terminate this span.
enum_body = re.search(r"^enum Commands \{\n(.*?)^\}", main_src, re.M | re.S)
if not enum_body:
    sys.exit(
        f"parse self-test failed: no `enum Commands {{` block in {main_path}; "
        "the CLI was restructured, so fix this guard before trusting it"
    )

# Variant-level attributes sit at the same 4-space indent as the variant. Field
# attributes inside a body are indented deeper and never match.
ATTR = re.compile(r"^ {4}#\[command\((.*)\)\]$")
# All three variant shapes, because missing one is silent rather than loud: an
# unmatched variant is simply never required to appear in the table.
#
#     Clean,            unit          -> ends after the comma (or at EOF)
#     Run {             struct        -> body follows on deeper-indented lines
#     Profile(String),  tuple         -> fields follow on the same line
#
# The 4-space indent is what separates variants from everything else in the
# block: doc comments start with `/`, attributes with `#`, a body's closing brace
# with `}`, and a variant's own fields are indented deeper. So the identifier
# only has to be followed by one of `{`, `(`, `,` or end of line -- not by
# nothing, which is what skipped the tuple form.
VARIANT = re.compile(r"^ {4}([A-Z][A-Za-z0-9]*)\s*(?:[{(,]|$)")

commands = []
hidden = []
pending = ""
for line in enum_body.group(1).splitlines():
    attr = ATTR.match(line)
    if attr:
        pending += attr.group(1)
        continue
    variant = VARIANT.match(line)
    if not variant:
        # Doc comments, blank lines and body content: keep any pending
        # attribute, since a doc comment may sit between it and the variant.
        if line.strip() and not line.lstrip().startswith("///"):
            if not line.startswith(" " * 8) and line.strip() not in ("},", "}"):
                pending = ""
        continue

    name_override = re.search(r'name\s*=\s*"([^"]+)"', pending)
    is_hidden = re.search(r"hide\s*=\s*true", pending) is not None
    pending = ""

    if name_override:
        name = name_override.group(1)
    else:
        # CamelCase -> kebab-case: EmitLtoir -> emit-ltoir.
        name = re.sub(r"(?<!^)(?=[A-Z])", "-", variant.group(1)).lower()

    (hidden if is_hidden else commands).append(name)

if len(commands) < 10:
    sys.exit(
        f"parse self-test failed: read {len(commands)} visible subcommands "
        f"(plus {len(hidden)} hidden) from {main_path}"
    )

# Only the Command reference section counts. installation.md carries several
# other tables (prerequisites, toolkit versions); a name appearing in one of
# those is not an entry in the CLI reference.
section = re.search(
    r"^### Command reference$(.*?)(?=^#{2,3} )", book_src, re.M | re.S
)
if not section:
    sys.exit(
        f"parse self-test failed: no `### Command reference` section in "
        f"{book_path}; it was renamed or removed, so fix this guard"
    )

# Leading underscores are matched on purpose: the hidden commands are named
# `__like-this`, and a row documenting one has to be reported rather than
# skipped as unrecognised text.
listed = re.findall(r"^\|\s*`([a-z_][a-z0-9_-]*)`\s*\|", section.group(1), re.M)
if len(listed) < 10:
    sys.exit(
        f"parse self-test failed: read {len(listed)} rows from the Command "
        f"reference in {book_path}"
    )

missing = [c for c in commands if c not in listed]
unknown = [row for row in listed if row not in commands]

failures = []
if missing:
    failures.append(
        "these subcommands have no row in the Command reference:\n"
        + "".join(f"    {c}\n" for c in missing)
    )
if unknown:
    detail = ""
    for row in unknown:
        why = " (hidden from --help)" if row in hidden else " (no such subcommand)"
        detail += f"    {row}{why}\n"
    failures.append(
        "the Command reference lists commands the CLI does not expose:\n" + detail
    )

if failures:
    print(f"error: {book_path} is out of step with {main_path}", file=sys.stderr)
    for failure in failures:
        print("  " + failure.rstrip(), file=sys.stderr)
    print(file=sys.stderr)
    print(
        "The table is the only complete list of the CLI in the docs. Add or "
        "remove the row,\ncopying the layout from a neighbour; the description "
        "column is clap's own short help.",
        file=sys.stderr,
    )
    sys.exit(1)

print(
    f"OK: {book_path} documents all {len(commands)} `cargo oxide` subcommands "
    f"({len(hidden)} hidden one(s) correctly excluded)."
)
PY
