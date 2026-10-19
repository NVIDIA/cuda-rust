#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Verify every copy of the toolchain pin still agrees with CUDA Oxide's
# rust-toolchain.toml.
#
# The pin is copied into several places, and nothing else checks that the
# copies move together:
#
#   1. The RUST_TOOLCHAIN_TOML scaffold in
#      crates/cargo-oxide/src/commands/scaffold.rs.
#      `cargo oxide new` writes it into every new project as that project's
#      rust-toolchain.toml, so this is the highest-impact copy: a stale
#      scaffold never breaks this repo's CI, it hands each new user a pin
#      whose backend cannot load, and the failure surfaces on their machine.
#
#   2. The rust feature in .devcontainer/devcontainer.json.  It preinstalls
#      the toolchain so the container's first build does not download it; a
#      stale version there warms the wrong cache, and a component the pin no
#      longer names keeps being installed into every container.
#
#   3. The `[toolchain]` blocks quoted in the book.  These are presented as
#      the repo's actual file, so a reader copies them into their own project.
#      When they go
#      stale the book hands out a pin that silently omits a component, and the
#      symptom lands later and elsewhere: a missing `rustfmt` surfaces as
#      `cargo oxide fmt` failing, not as a bad toolchain file.
#
#   4. The dated commands and prose across the book and the READMEs:
#      `rustup toolchain install nightly-...`, `cargo +nightly-... install`,
#      and sentences naming the pin.  Readers run those commands outside a
#      checkout, where no rust-toolchain.toml can correct a stale date.
#
# This is exactly what happened: #727 added `rustfmt` to both real files and
# left both book quotes at five components.
#
# Channel and components only.  Anything else in a [toolchain] block (a
# `targets` list, `profile`) is free to differ between the copies, and the
# book quotes elide freely -- the check is that what a copy *does* state about
# these two keys is right, not that it restates everything.
#
# Run this after bumping the pin or adding a component.
set -euo pipefail

# Both python and the comparisons below are byte-wise; pin the locale so an
# ambient UTF-8 one cannot reorder a components list under dictionary
# collation and turn an agreeing pair into a spurious failure.
export LC_ALL=C

cd "$(dirname "$0")/.."

ROOT_PIN=rust-toolchain.toml
SCAFFOLD=crates/cargo-oxide/src/commands/scaffold.rs
DEVCONTAINER=.devcontainer/devcontainer.json

if ! command -v python3 >/dev/null 2>&1; then
    echo "error: python3 is required to verify the toolchain pin" >&2
    echo "       refusing to report success from a check that cannot run" >&2
    exit 1
fi

# `git ls-files` is the only precise notion of "tracked markdown": a bare
# glob would also sweep up untracked trees (a local target/ holds registry
# READMEs) and flag files that are not this repo's to keep consistent.
if ! command -v git >/dev/null 2>&1; then
    echo "error: git is required to enumerate the tracked markdown" >&2
    echo "       refusing to report success from a check that cannot run" >&2
    exit 1
fi

test -s "${ROOT_PIN}"
test -s "${SCAFFOLD}"
test -s "${DEVCONTAINER}"

GIT_ROOT="$(git rev-parse --show-toplevel)"
python3 - "${ROOT_PIN}" "${SCAFFOLD}" "${DEVCONTAINER}" "${GIT_ROOT}" <<'PY'
import glob
import json
import os
import re
import subprocess
import sys

root_path, scaffold_path, devcontainer_path, git_root = sys.argv[1:5]

CHANNEL = re.compile(r'^\s*channel\s*=\s*"([^"]+)"', re.M)
# Match the whole bracketed span and pull the quoted names out of it so
# reflowing the list never breaks this guard.
COMPONENTS = re.compile(r"^\s*components\s*=\s*\[(.*?)\]", re.M | re.S)


def read(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def pin(text, source):
    """(channel, components) from a [toolchain] block; None where unstated."""
    channel = CHANNEL.search(text)
    components = COMPONENTS.search(text)
    return (
        channel.group(1) if channel else None,
        re.findall(r'"([^"]+)"', components.group(1)) if components else None,
    )


root_channel, root_components = pin(read(root_path), root_path)

# Parse self-test.  A guard whose failure mode is "matched nothing" has to
# prove it still reads its reference before a clean result means anything.
if not root_channel or not root_components:
    sys.exit(
        f"parse self-test failed: read channel={root_channel!r} "
        f"components={root_components!r} from {root_path}; "
        "the file layout changed, fix this script before trusting it"
    )

failures = []

# The scaffold `cargo oxide new` writes into user projects.  It is a full
# rust-toolchain.toml, so both keys are required and must match exactly.
SCAFFOLD_CONST = re.compile(
    r'^const RUST_TOOLCHAIN_TOML: &str = r#"(.*?)"#;', re.M | re.S
)
scaffold_const = SCAFFOLD_CONST.search(read(scaffold_path))
if not scaffold_const:
    sys.exit(
        "parse self-test failed: no RUST_TOOLCHAIN_TOML raw-string const "
        f"found in {scaffold_path}; the constant moved or was renamed, fix "
        "this script before trusting it"
    )
scaffold_channel, scaffold_components = pin(scaffold_const.group(1), scaffold_path)
if not scaffold_channel or not scaffold_components:
    sys.exit(
        f"parse self-test failed: read channel={scaffold_channel!r} "
        f"components={scaffold_components!r} from the RUST_TOOLCHAIN_TOML "
        f"const in {scaffold_path}; the scaffold layout changed, fix this "
        "script before trusting it"
    )
if scaffold_channel != root_channel:
    failures.append(
        f"{scaffold_path} scaffolds channel {scaffold_channel!r}, "
        f"{root_path} pins {root_channel!r}"
    )
if scaffold_components != root_components:
    failures.append(
        f"{scaffold_path} scaffolds components {scaffold_components!r}, "
        f"{root_path} lists {root_components!r}"
    )

# The devcontainer's preinstalled toolchain.  The channel must be the pin's.
# The components line is a deliberate warm-cache subset -- the container
# ships its own LLVM 21 and sets CUDA_OXIDE_LLC, so rustup's llvm-tools is
# left for rust-toolchain.toml to pull on first use -- but everything it
# *does* preinstall must be a component the pin still names, or a pin change
# leaves every new container installing a leftover.
try:
    devcontainer = json.loads(read(devcontainer_path))
except ValueError as error:
    sys.exit(
        f"parse self-test failed: {devcontainer_path} is not plain JSON "
        f"({error}); if it grew JSONC comments, fix this script before "
        "trusting it"
    )
# Feature keys carry a version tag (".../rust:1"); strip it and require an
# exact id so a renamed feature trips the self-test instead of a lookalike
# (".../rustup", ".../rust-lang") being read as the rust feature.
rust_features = [
    value
    for key, value in devcontainer.get("features", {}).items()
    if key.split(":")[0] == "ghcr.io/devcontainers/features/rust"
]
if len(rust_features) != 1 or "version" not in rust_features[0]:
    sys.exit(
        "parse self-test failed: expected one rust feature with a version "
        f"in {devcontainer_path}, found {len(rust_features)}; the feature "
        "moved or was renamed, fix this script before trusting it"
    )
devcontainer_channel = rust_features[0]["version"]
if devcontainer_channel != root_channel:
    failures.append(
        f"{devcontainer_path} preinstalls {devcontainer_channel!r}, "
        f"{root_path} pins {root_channel!r}"
    )
devcontainer_components = [
    name for name in rust_features[0].get("components", "").split(",") if name
]
stale = [name for name in devcontainer_components if name not in root_components]
if stale:
    failures.append(
        f"{devcontainer_path} preinstalls component(s) the pin does not "
        f"name: {' '.join(stale)}; {root_path} lists {root_components!r}"
    )

# The book's quoted blocks.  Only fenced ```toml blocks that actually contain
# a [toolchain] header are candidates: prose that merely names the file, and
# the `rustup component add` command lines (which are deliberately not
# exhaustive -- rustup installs whatever the pin names), are not quotes of it.
docs = sorted(glob.glob("cuda-oxide-book/**/*.md", recursive=True))
if len(docs) < 20:
    sys.exit(f"parse self-test failed: found {len(docs)} book pages")

TOML_BLOCK = re.compile(r"```toml\n(.*?)```", re.S)
quoted = 0
for path in docs:
    for block in TOML_BLOCK.findall(read(path)):
        if "[toolchain]" not in block:
            continue
        quoted += 1
        channel, components = pin(block, path)
        if channel is not None and channel != root_channel:
            failures.append(
                f"{path} quotes channel {channel!r}, {root_path} pins {root_channel!r}"
            )
        if components is not None and components != root_components:
            missing = [c for c in root_components if c not in components]
            extra = [c for c in components if c not in root_components]
            detail = ", ".join(
                part
                for part in (
                    "missing " + " ".join(missing) if missing else "",
                    "unknown " + " ".join(extra) if extra else "",
                )
                if part
            )
            failures.append(
                f"{path} quotes components {components!r}"
                + (f" ({detail})" if detail else "")
                + f"; {root_path} lists {root_components!r}"
            )

if not quoted:
    sys.exit(
        "parse self-test failed: no [toolchain] block found in any book page; "
        "either the book stopped quoting the pin (delete that half of this "
        "guard) or the fence style changed (fix the pattern)"
    )

# Every dated nightly reference in tracked markdown.  The book's install
# pages and the READMEs spell the pin inside commands a reader runs outside
# a checkout (`rustup toolchain install nightly-...`,
# `cargo +nightly-... install`), where no rust-toolchain.toml can correct a
# stale date, plus prose naming the pin.  The token itself is the check --
# any dated nightly a tracked page mentions must be the pinned one -- so the
# guard survives pages being reworded, moved, or added.  Component lists on
# `rustup component add` lines stay uncovered, as above; the channel date on
# those same lines is checked like any other.
DATED = re.compile(r"nightly-\d{4}-\d{2}-\d{2}")
if not DATED.fullmatch(root_channel):
    sys.exit(
        f"parse self-test failed: {root_path} channel {root_channel!r} is "
        "not a dated nightly; this guard assumes an exact pin"
    )
markdown = sorted(
    path
    for path in subprocess.run(
        ["git", "-C", git_root, "ls-files", "-z", "--", "*.md"],
        stdout=subprocess.PIPE,
        check=True,
    )
    .stdout.decode()
    .split("\0")
    if path
)
if len(markdown) < 20:
    sys.exit(
        f"parse self-test failed: git lists {len(markdown)} tracked "
        "markdown files"
    )

dated = 0
for path in markdown:
    for number, line in enumerate(read(os.path.join(git_root, path)).splitlines(), start=1):
        for token in DATED.findall(line):
            dated += 1
            if token != root_channel:
                failures.append(
                    f"{path}:{number} spells {token!r}, "
                    f"{root_path} pins {root_channel!r}"
                )

if not dated:
    sys.exit(
        "parse self-test failed: no dated nightly reference found in any "
        "tracked markdown; either the docs stopped spelling the pin (delete "
        "this block of the guard) or the pattern is stale (fix it)"
    )

if failures:
    print("error: the toolchain pin is not consistent across its copies", file=sys.stderr)
    for failure in failures:
        print(f"  {failure}", file=sys.stderr)
    print(file=sys.stderr)
    print(
        f"Every copy must state the same channel and components as {root_path}. "
        "The `cargo oxide new` scaffold must match exactly "
        "(rustc_private dylibs);\nthe book's blocks are presented to readers as "
        "the repo's real file, and the\ndated commands run where no "
        "rust-toolchain.toml can correct them.",
        file=sys.stderr,
    )
    sys.exit(1)

print(
    f"OK: {root_path} pins {root_channel} with {len(root_components)} components; "
    "the `cargo oxide new` scaffold, the devcontainer, all "
    f"{quoted} block(s) quoted in the book, and all {dated} dated reference(s) "
    "in tracked markdown agree."
)
PY
