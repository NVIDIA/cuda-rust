#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Verify every device-API name the docs show resolves to a function
# `cuda-device` actually exports.
#
# Scope is the book plus every crate and example README. The book half is the
# original; the READMEs were added after one of them was found naming
# `warp::shfl`, which has never existed. `cuda-device` exports 32 `shuffle*`
# functions and no `shfl`; `shfl` is the cooperative-groups *method* name.
# Example READMEs carry over a hundred qualified device names and are
# the first thing a reader of an example opens, so they rot the same way the
# book does and were checked by nothing.
#
# The failure this catches is silent and has now happened three times. #797
# found the intrinsics guide pointing at op files that no longer existed; the
# same sweep later found `warp::shuffle_xor_i32` in the API quick reference,
# advertised as an "i32 variant" that has never existed, and a dispatch helper
# in the compiler pages that had been replaced by generated code. Nothing fails
# when a page names a function the tree does not have: it renders, it builds,
# and a reader finds out by pasting it.
#
# Two passes, because the docs name APIs two ways:
#
#   * A call in a ```rust fenced block -- `warp::foo(...)` -- which a reader
#     pastes, so the name has to exist.
#   * A backticked, module-qualified name in prose or a table, with or without a
#     following `(`. This pass is what #815 needed and the first pass would have
#     missed: that fix removed one code-block line, `warp::shuffle_xor_i32(...)`,
#     *and* four table rows written `shuffle_xor_{f32,i32}(val, mask)`. Only the
#     first is a call.
#
# Backticks are what keep the second pass precise. Restricting to code blocks
# was originally about avoiding matches on a PTX mnemonic (`shfl.sync.bfly`), a
# module path, or another project's function in running text. A backticked,
# module-qualified name is a claim about this API wherever it appears, so it is
# safe to check; unqualified table entries stay out of scope, since deciding
# which module a bare `shuffle_xor` belongs to is guesswork.
#
# Scope otherwise unchanged, so the guard stays precise rather than merely broad:
#
#   * Only the device modules -- `warp`, `thread`, `grid`, `cluster`. Those are
#     the paths the docs use in kernel examples and the ones that rot. Host
#     APIs are checked by rustdoc, which builds under `-D warnings`.
#   * Existence only, never arity or types. Those change for good reasons and
#     the compiler catches them; a name that is simply absent is the silent case.
#
# Run this after renaming or removing anything in `cuda-device`.
set -euo pipefail

export LC_ALL=C

cd "$(dirname "$0")/.."

BOOK=cuda-oxide-book
DEVICE=crates/cuda-device/src
# Host crates live at the git root; SIMT crates live here.
CRATE_ROOTS=(../cuda-core ../cuda-bindings ../cuda-async crates)

if ! command -v python3 >/dev/null 2>&1; then
    echo "error: python3 is required to verify the docs' API names" >&2
    echo "       refusing to report success from a check that cannot run" >&2
    exit 1
fi

test -d "${BOOK}"
test -d "${DEVICE}"
for root in "${CRATE_ROOTS[@]}"; do
    test -d "${root}"
done

python3 - "${BOOK}" "${DEVICE}" "${CRATE_ROOTS[@]}" <<'PY'
import glob
import os
import re
import sys

book_root, device_root, *crates_roots = sys.argv[1:]

RUST_BLOCK = re.compile(r"```rust[^\n]*\n(.*?)```", re.S)
# `mod::name(` inside a code block: a call, so the name must exist.
#
# The lookbehind is load-bearing. `warp` and `thread` are also module names deep
# inside the compiler -- the intrinsics guide legitimately shows
# `intrinsics::warp::emit_two_operand_intrinsic(...)`, a mir-importer path that
# has no business resolving against cuda-device. Rejecting a `::` immediately
# before the module keeps this to the device paths a kernel actually calls,
# while an explicit `cuda_device::` prefix stays accepted.
CALL = re.compile(
    r"(?<!::)\b(?:cuda_device::)?(warp|thread|grid|cluster)::([A-Za-z_][A-Za-z0-9_]*)\s*\("
)
EXPORTED = re.compile(r"pub (?:unsafe )?fn ([A-Za-z_][A-Za-z0-9_]*)")

# The same qualified names in prose or a table. The brace group deliberately does
# not require its own leading `_`: the identifier class is greedy and would
# otherwise swallow the `_` in `threadIdx_{x,y,z}`, leaving `{x,y,z}` unmatched
# and the truncated `threadIdx_` reported as missing.
PROSE_NAME = re.compile(
    r"`(?:cuda_device::)?(warp|thread|grid|cluster)::"
    r"([A-Za-z_][A-Za-z0-9_]*(?:\{[A-Za-z0-9_,]+\}[A-Za-z0-9_]*)?)"
)
# `name_{a,b,c}` is the shorthand for a family, live today in
# `thread::threadIdx_{x,y,z}()` and the notation that carried #815's fictional
# rows. Expanded, so a family with one bad member fails on that member instead of
# passing as an unrecognised literal.
#
# The group can sit anywhere in the name, not only at the end:
# `warp::reduce_{sum,max,min}_f32` is real notation in an example README, and an
# end-anchored pattern read it as the family `reduce_{sum,max,min}` and dropped
# the `_f32`, reporting three functions missing that were never named.
BRACES = re.compile(r"^([A-Za-z0-9_]*)\{([A-Za-z0-9_,]+)\}([A-Za-z0-9_]*)$")


def expand(name):
    """`foo_{a,b}_bar` -> [`foo_a_bar`, `foo_b_bar`]; anything else -> [itself]."""
    match = BRACES.match(name)
    if not match:
        return [name]
    prefix, options, suffix = match.group(1), match.group(2), match.group(3)
    return [
        f"{prefix}{option.strip()}{suffix}"
        for option in options.split(",")
        if option.strip()
    ]


pages = sorted(glob.glob(os.path.join(book_root, "**", "*.md"), recursive=True))
if len(pages) < 20:
    sys.exit(f"parse self-test failed: found {len(pages)} book pages under {book_root}")

# Crate and example READMEs, minus the vendored rustlantis subtree, whose files
# keep their upstream form and document another project's API.
readmes = sorted(
    path
    for crates_root in crates_roots
    for path in glob.glob(os.path.join(crates_root, "**", "README.md"), recursive=True)
    if "rustlantis" not in path.split(os.sep)
)
if len(readmes) < 50:
    sys.exit(
        f"parse self-test failed: found {len(readmes)} READMEs under "
        + ", ".join(crates_roots)
    )
pages += readmes

calls = {}
mentions = {}
blocks = 0
for page in pages:
    with open(page, encoding="utf-8") as handle:
        text = handle.read()
    for block in RUST_BLOCK.findall(text):
        blocks += 1
        for match in CALL.finditer(block):
            calls.setdefault((match.group(1), match.group(2)), set()).add(page)

    # Everything outside a fenced block, so each name is counted once by
    # whichever pass owns it.
    prose = re.sub(r"```.*?```", "", text, flags=re.S)
    for match in PROSE_NAME.finditer(prose):
        for name in expand(match.group(2)):
            mentions.setdefault((match.group(1), name), set()).add(page)

if blocks < 20:
    sys.exit(
        f"parse self-test failed: read {blocks} rust code blocks from the book "
        "and READMEs"
    )

# Both passes have to find something, or a rot in either regex reads as a clean
# tree rather than a broken check.
if len(mentions) < 5:
    sys.exit(
        f"parse self-test failed: found {len(mentions)} qualified device names in "
        "the docs' prose and tables"
    )

exported = set()
sources = sorted(glob.glob(os.path.join(device_root, "**", "*.rs"), recursive=True))
for source in sources:
    with open(source, encoding="utf-8") as handle:
        exported |= set(EXPORTED.findall(handle.read()))

# The other half of the silent-blindness guard: if the export scan rotted, every
# name would look missing rather than every name looking fine, but say so
# explicitly rather than dumping hundreds of failures.
if len(exported) < 200:
    sys.exit(
        f"parse self-test failed: read {len(exported)} exported fns from "
        f"{len(sources)} files under {device_root}"
    )


def unresolved(found):
    return sorted(
        (module, name, sorted(where))
        for (module, name), where in found.items()
        if name not in exported
    )


def report(found, heading, closing):
    if not found:
        return False
    print(heading, file=sys.stderr)
    for module, name, where in found:
        pages_text = ", ".join(os.path.relpath(p) for p in where)
        print(f"  {module}::{name}   in {pages_text}", file=sys.stderr)
    print(file=sys.stderr)
    print(closing, file=sys.stderr)
    return True


failed = report(
    unresolved(calls),
    "error: the docs call device functions that cuda-device does not export:",
    "A Rust code block is something a reader pastes. Either the function was\n"
    "renamed and the docs missed it, or the example was written from memory.",
)
failed = (
    report(
        unresolved(mentions),
        "error: the docs name device functions that cuda-device does not export:",
        "These are in prose or a table rather than a code block, which is where the\n"
        "fictional `shuffle_xor_{f32,i32}` rows survived until #815. A `_{a,b}`\n"
        "family is expanded, so the member named above is the one that is missing.",
    )
    or failed
)

if failed:
    sys.exit(1)

print(
    f"OK: all {len(calls)} device-API calls in {blocks} Rust blocks across the "
    f"book and {len(readmes)} READMEs, and "
    f"{len(mentions)} qualified names in their prose and tables, resolve against "
    f"cuda-device's {len(exported)} exported functions."
)
PY
