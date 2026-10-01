#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Enforce deny.toml over the example workspaces, which `cargo deny check` does
# not reach.
#
# `cargo deny check` resolves the root workspace.  Every example under
# crates/rustc-codegen-cuda/examples/ sets its own `[workspace]`, so the root
# run stops at that boundary and the license, source and ban policies never see
# any crate an example pulls on its own.  #664 closed the *inventory* half of
# this (dependency-licenses.csv now records those crates); this closes the
# *policy* half, so the allow-list in deny.toml actually governs them.
#
# Measured when this guard was written: 186 of the 187 example lock files
# already satisfy the policy unchanged, so this is a guard against future drift
# rather than a fix for a present violation.  The one exception is exempted
# below, with its reason.
#
# Why one run per distinct dependency set, not one per workspace:
#
#   Every example depends on cuda-device/cuda-host by path and on the shared
#   cuda-core by the same cutile-rs pin as the root workspace, so each lock
#   file re-lists the root workspace's own transitive crates.  Grouping the
#   lock files by their exact set of third-party (name, version, source) triples
#   collapses every example lock file (one per example directory, plus one per
#   nested sub-workspace) to far fewer distinct sets, and one cargo-deny run
#   each.  The run prints both counts, so this comment does not repeat them.
#   The grouping is an equivalence, not a sample: license, source and advisory
#   verdicts are per-crate properties, so two workspaces resolving the
#   identical crate set get the identical verdict.  `bans.multiple-versions`
#   is a graph property, but deny.toml sets it to "warn", so it cannot change
#   the exit status.
set -euo pipefail

export LC_ALL=C

cd "$(dirname "$0")/.."

EXAMPLES_ROOT=crates/rustc-codegen-cuda/examples

# Examples whose dependencies deliberately cannot satisfy deny.toml.
#
# None today.  cutile_inter_kernel used to be exempt (#953): it linked
# NVlabs/cutile-rs by git while `[sources] allow-git` listed only pliron, and
# cutile-rs's cuda-bindings 0.1.0 carried no license.  Both are gone: every
# example now takes the shared host crates from the same crates.io release as
# the root workspace, and the 0.3.x crates are Apache-2.0, so the example is
# judged like every other one.
#
# Every name here is checked against the examples on disk below, so a typo or a
# rename fails the run instead of quietly exempting nothing -- or everything.
# An exemption covers every lock file under the example, nested sub-workspaces
# included.
POLICY_EXEMPT_EXAMPLES=()

command -v cargo-deny >/dev/null 2>&1 || {
    echo "error: cargo-deny not found; install it with 'cargo install cargo-deny --locked'" >&2
    exit 1
}

# One representative manifest per distinct third-party dependency set.  Lock
# files, not example directories, are the unit: a nested sub-workspace resolves
# its own graph, so it must be checked (or exempted) as a workspace of its own,
# not merely folded into its parent's grouping key.
representatives="$(python3 -c '
import glob, os, re, sys

examples_root, *exempt = sys.argv[1:]

def third_party(lock):
    """(name, version, source) for every locked package that has a source.

    A package with no `source` is a path dependency, i.e. first-party by
    construction, and carries no policy question of its own.  The source is
    part of the key because the sources policy judges where a crate comes
    from: the same (name, version) from crates.io and from a git fork are
    different policy questions.
    """
    found = set()
    parsed = 0
    for block in open(lock).read().split("[[package]]")[1:]:
        name = re.search(r"^name = \"([^\"]+)\"", block, re.M)
        version = re.search(r"^version = \"([^\"]+)\"", block, re.M)
        source = re.search(r"^source = \"([^\"]+)\"", block, re.M)
        if name and version:
            # Counted after the match, not per `[[package]]` split: every
            # package carries a name and a version, so a regex that stops
            # matching takes this to zero.  Counting the splits instead made
            # the tally below blind to exactly the rot it guards against.
            parsed += 1
        if name and version and source:
            found.add((name.group(1), version.group(1), source.group(1)))
    return found, parsed

locks = sorted(glob.glob(os.path.join(examples_root, "**", "Cargo.lock"), recursive=True))
on_disk = {os.path.relpath(lock, examples_root).split(os.sep)[0] for lock in locks}

# Cross-check the glob against an independent enumeration rather than a fixed
# floor.  Every example directory that carries a Cargo.toml carries a
# Cargo.lock beside it, plus one per nested sub-workspace, so a name absent
# here means the glob stopped reaching it.  A count cannot tell a narrowed
# glob from a smaller tree: measured once, a glob narrowed to `[a-m]*` still
# found 137 of the 217 locks then present, which clears any floor loose enough
# not to fail on ordinary growth.  The floor this replaces was 20.
expected = {
    name
    for name in os.listdir(examples_root)
    if os.path.exists(os.path.join(examples_root, name, "Cargo.toml"))
}
missed = sorted(expected - on_disk)
if missed:
    sys.exit(
        "found no Cargo.lock for %d example(s), so deny.toml cannot be "
        "enforced over them: %s.  Commit the lock file, or fix the glob above "
        "if it stopped reaching them." % (len(missed), ", ".join(missed[:5]))
    )

unknown = sorted(set(exempt) - on_disk)
if unknown:
    sys.exit("POLICY_EXEMPT_EXAMPLES names no such example: " + ", ".join(unknown))

groups = {}
total_parsed = 0
for lock in locks:
    example = os.path.relpath(lock, examples_root).split(os.sep)[0]
    crates, parsed = third_party(lock)
    total_parsed += parsed
    # Every example reaches crates.io through cuda-core/cuda-device/cuda-host,
    # so a lock resolving no third-party crate at all means the parse failed,
    # not that the example is dependency-free.  Measured range across the tree:
    # 46 to 142 third-party crates per lock, so this has wide margin -- and it
    # goes to zero for every lock the moment a regex rots, which is the case
    # the tally below is meant to catch and could not.
    if not crates:
        sys.exit("parse self-test failed: no third-party crates in %s" % lock)
    if example in exempt:
        continue
    groups.setdefault(frozenset(crates), os.path.join(os.path.dirname(lock), "Cargo.toml"))

# Mirrors the inventory guard: if the lock-file regexes silently rotted, the
# groups would quietly collapse and a single run would vouch for everything.
# Scaled to the locks actually found rather than a fixed number, so it cannot
# go stale as the tree grows: every lock yields dozens of parsed packages
# against a floor of ten (measured once: about 64 per lock), and a regex rot
# takes the tally to zero.
if total_parsed < 10 * len(locks):
    sys.exit("parse self-test failed: parsed %d packages from %d lock files"
             % (total_parsed, len(locks)))

for manifest in sorted(groups.values()):
    print(manifest)
' "${EXAMPLES_ROOT}" "${POLICY_EXEMPT_EXAMPLES[@]}")"

total="$(printf '%s\n' "${representatives}" | grep -c .)"
locks="$(find "${EXAMPLES_ROOT}" -name Cargo.lock | grep -c .)"
echo "Checking deny.toml over ${total} representative example workspaces" \
    "across ${locks} example lock files."

# No --config: cargo-deny resolves the config by walking up from the manifest
# directory, so every example workspace finds cuda-oxide/deny.toml.
# (Verified on cargo-deny 0.19 and 0.20; a top-level --config flag only exists
# on 0.20.)  --locked so a stale lock file fails the run instead of being
# silently re-resolved: the committed lock is what grouped the workspace, so it
# must be what the policy judges.
failed=()
for manifest in ${representatives}; do
    if ! cargo deny --manifest-path "${manifest}" --locked check 2>&1 |
        sed "s|^|  [$(dirname "${manifest#"${EXAMPLES_ROOT}/"}")] |"; then
        failed+=("${manifest}")
    fi
done

if ((${#failed[@]})); then
    echo "error: deny.toml is not satisfied by these example workspaces:" >&2
    printf '  %s\n' "${failed[@]}" >&2
    echo "       Each is the representative for a group of workspaces resolving the" >&2
    echo "       same third-party crates, so the cause is shared by its whole group." >&2
    exit 1
fi

echo "OK: deny.toml holds over every example workspace outside POLICY_EXEMPT_EXAMPLES."
