#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Reject a prose reference to a source file that is not in the repository.
#
# #1190 split large files into module directories and left three prose
# references pointing at paths it had removed -- a test's module doc at
# `crates/mir-lower/src/convert/types.rs`, two example READMEs at
# `crates/mir-importer/src/translator/rvalue.rs`. #1197 repointed those, and 14
# more in other forms. Nothing had noticed: a path in prose is compiled by
# nothing, `check-host-api-paths.sh` checks Rust *API* paths rather than files,
# and the book gate only builds. The reference just goes quiet, and a reader
# follows it into nothing.
#
# "In the repository" means **tracked by git**, not present on disk. Those are
# different questions and only the first one is reproducible: a path that
# exists solely as a local build artifact, or an untracked file someone has in
# their tree, resolves for them and for nobody else. Testing the filesystem
# would make this guard pass or fail depending on whose checkout it ran in.
#
# Scope, deliberately narrow, because this is the kind of guard that goes soft
# the moment it starts inferring:
#
#   * A path is only checked when it is anchored at a repo root --
#     cuda-oxide/crates/, cuda-oxide/cuda-oxide-book/, cuda-oxide/scripts/ --
#     and carries a file extension. That
#     is the form a new reference normally takes, and the only one that is
#     unambiguous on its own. Each root is verified to be a real tracked
#     directory before the sweep, so a renamed root fails here rather than
#     silently matching nothing.
#   * Prose only: every line of a tracked *.md, and `///` or `//!` lines in a
#     tracked *.rs. This is what removes the need for a general exemption
#     list. The non-existent paths that live in Rust *code* are all deliberate
#     -- synthetic fixture names (`intrinsics/probes/removed.ll`,
#     `intrinsics/overlay/test.toml`), two paths that cuda-intrinsics-gen
#     render tests assert are *absent*, and the mktemp canary in
#     check-reserved-prefixes.sh -- and none of them is prose.
#
# Generated outputs are declared, never inferred. An earlier revision skipped a
# path whose parent directory held no tracked file, reasoning that such a
# directory must be an output location. That is exactly backwards: deleting or
# mistyping a directory produces the same signal as generating into one, so
# the broken reference this guard exists to catch was the case it let through.
# The list below is the whole accommodation, and it is verified rather than
# trusted -- an entry nothing refers to any more is an error, so it cannot rot
# into a silent exemption.
#
# Two things stay out of scope on purpose:
#
#   * `intrinsics/` is not a root. It is both a directory under cuda-oxide/ and a
#     common crate-relative fragment: examples/atomics/README.md names
#     `intrinsics/atomic.rs` in a pipeline diagram, meaning
#     `mir-importer/src/translator/terminator/intrinsics/atomic.rs`, which is
#     correct as shorthand. Rooting there would fail that line.
#   * Crate-relative (`mir-lower/src/convert/types.rs`) and bare-basename
#     ("the walker in `rvalue.rs`") forms are not checked. #1197's
#     follow-up fixed 14 references written that way, so this is a real gap and
#     is stated rather than papered over -- telling `rvalue.rs` in prose from
#     any other mention of it is guesswork, and a guard that guesses is worse
#     than one that is narrow.
#   * Prose under `cutile-rs/` is written for that tree's root. A `scripts/`
#     reference there means `cutile-rs/scripts/`. Prose under `cuda-oxide/` is
#     written for that tree: `scripts/`, `crates/`, and `cuda-oxide-book/`
#     there mean the same paths with a `cuda-oxide/` prefix. Only those
#     prefixes are rewritten, and only for files already in that tree. A
#     `scripts/run_all.sh` reference outside both trees still fails.
set -euo pipefail
export LC_ALL=C
cd "$(git rev-parse --show-toplevel)"

if ! command -v python3 >/dev/null 2>&1; then
    echo "error: python3 is required to check source references" >&2
    exit 1
fi

python3 - <<'PY'
import pathlib
import re
import subprocess
import sys
import tempfile

ROOTS = ("cuda-oxide/crates", "cuda-oxide/cuda-oxide-book", "cuda-oxide/scripts")
# Short anchors are how prose inside a product tree names that tree.
# checked_references adds the product prefix; these are not tracked roots.
PATH_ROOTS = ROOTS + ("crates", "cuda-oxide-book", "scripts")
EXTS = ("rs", "md", "sh", "toml", "jsonl", "json", "ll", "py", "yaml", "yml")
# run_seed.py writes and clears this output; the fuzzer README names it.
GENERATED_OUTPUT_PATHS = {"cuda-oxide/crates/fuzzer/artifacts/summary.jsonl"}

# Read complete tokens first, then classify them. Whitespace and ordinary
# Markdown/prose wrappers delimit tokens. Scheme-prefixed and network-path
# URI spans are consumed first, including their internal prose punctuation;
# `https://host/x;crates/foo.rs` is one external reference. Balanced square
# brackets stay within a URI (e.g. an IPv6 host); an unmatched closing bracket
# can still end a Markdown link label. A slash, dot, hyphen, backslash or
# colon stays inside other tokens, so longer paths never donate suffixes.
# This is lexical classification, not a URL resolver or a scheme allowlist.
TOKEN = re.compile(
    r'''(?P<uri>[*_]*(?:[A-Za-z][A-Za-z0-9+.-]*:|//)(?:\[[^\]\s`'"<>]*\]|[^\s`'"<>\[\]])+)'''
    r'''|[^\s`'"<>()\[\],;|]+'''
)
PATH = re.compile(
    r"(?:" + "|".join(map(re.escape, PATH_ROOTS)) + r")/[A-Za-z0-9._/-]+\."
    r"(?:" + "|".join(map(re.escape, EXTS)) + r")"
)
DOC_LINE = re.compile(r"^\s*(?:///(?!/)|//!)")
LINE_SUFFIX = re.compile(r":[0-9]+(?:[-:][0-9]+)*$")
# A Markdown reference definition owns its colon; it is not part of the
# destination token. Strip only the line's label prefix, before classifying
# the destination, so a URI (including a custom scheme) stays intact.
REFERENCE_DEFINITION = re.compile(r"^ {0,3}\[(?:\\.|[^\[\]\\])+\]:[ \t]*")


def paths_in_line(line):
    definition = REFERENCE_DEFINITION.match(line)
    cursor = definition.end() if definition else 0
    while match := TOKEN.search(line, cursor):
        cursor = match.end()
        if match.lastgroup == "uri":
            # A parenthesized URI ends at its matching closing delimiter,
            # not at the end of the next adjacent Markdown link. Balance
            # nested parentheses in the URI and preserve escaped delimiters.
            # Bare/angle-quoted URIs retain their entire punctuation span.
            if match.start() > 0 and line[match.start() - 1] == "(":
                depth = 0
                escaped = False
                for index in range(match.start(), match.end()):
                    char = line[index]
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == "(":
                        depth += 1
                    elif char == ")":
                        if depth == 0:
                            cursor = index + 1
                            break
                        depth -= 1
            continue
        token = match.group().rstrip(".!?:")
        # Paired, possibly nested Markdown emphasis is a wrapper, not a glob
        # suffix. Punctuation can sit inside or outside the closing marker.
        while len(token) > 1 and token[0] in "*_" and token[-1] == token[0]:
            token = token[1:-1]
        token = token.rstrip(".!?:")
        # A link fragment and a numeric source-line/range citation are not
        # part of a filename. Other suffixes stay intact: `.rs.backup` must
        # not be checked as `.rs`, and `:word` is not a line citation.
        token = token.partition("#")[0]
        token = LINE_SUFFIX.sub("", token)
        # The tree also writes shell commands as ./scripts/foo.sh. Accept
        # that explicit current-root form, without resolving parents,
        # absolute paths or another project's leading directory.
        if token.startswith("./"):
            token = token[2:]
        if PATH.fullmatch(token):
            yield token


def references(filename, text):
    suffix = pathlib.PurePosixPath(filename).suffix
    for lineno, line in enumerate(text.splitlines(), 1):
        if suffix == ".rs":
            marker = DOC_LINE.match(line)
            if marker is None:
                continue
            line = line[marker.end():]
        if suffix in (".md", ".rs"):
            for path in sorted(set(paths_in_line(line))):
                yield lineno, path


def checked_references(filename, text):
    # See the header. Rewrite only the anchors that are relative to a product tree.
    for lineno, path in references(filename, text):
        if filename.startswith("cutile-rs/") and path.startswith("scripts/"):
            path = "cutile-rs/" + path
        elif filename.startswith("cuda-oxide/") and (
            path.startswith("scripts/")
            or path.startswith("crates/")
            or path.startswith("cuda-oxide-book/")
        ):
            path = "cuda-oxide/" + path
        yield lineno, path


def broken_references(refs, tracked, generated):
    for lineno, path in refs:
        if ".." in path.split("/"):
            yield lineno, path, "contains an unsupported traversal segment"
        elif path not in generated and path not in tracked:
            yield lineno, path, "is not tracked in this repository"


def declaration_errors(generated, tracked, referenced):
    for path in sorted(generated):
        if path in tracked:
            yield f"'{path}' is declared generated but is tracked"
        elif path not in referenced:
            yield f"no scanned prose reference names generated output '{path}'"


tracked = set(subprocess.check_output(["git", "ls-files", "-z"]).decode().split("\0"))
tracked.discard("")
for root in ROOTS:
    if not any(path.startswith(root + "/") for path in tracked):
        sys.exit(f"error: source-reference guard: '{root}/' holds no tracked file; update ROOTS")

# Exercise the actual extractor, prose filter, and membership decision. The
# untracked control really exists on disk, so replacing index membership with
# filesystem existence fails here. The tracked example comes from this index,
# which also permits running the guard in historical worktrees.
live = next((path for path in sorted(tracked) if PATH.fullmatch(path)), None)
if live is None:
    sys.exit("error: source-reference guard: no tracked path matches the declared roots/extensions")
with tempfile.NamedTemporaryFile(prefix="zz-source-reference-", suffix=".rs", dir="cuda-oxide/crates") as canary:
    untracked = pathlib.Path(canary.name).relative_to(pathlib.Path.cwd()).as_posix()
    missing = "cuda-oxide/crates/no-such-source-reference-crate/src/lib.rs"
    # Synthetic policy inputs exercise the same decision functions even when
    # the repository no longer needs any generated-output declarations.
    output = "cuda-oxide/crates/source-reference-canary/generated.jsonl"
    generated = {output}
    controls = [
        ("case.md", "cuda-oxide/crates/cuda-device/src/no-such-source-reference-file.rs", 1),
        ("case.md", missing, 1),
        ("case.md", untracked, 1),
        ("case.md", f"`{live}`, [{live}]({live}#L1); **{live}**; {live}:1-2.", 0),
        ("case.md", output, 0),
        ("case.md", f"https://example.invalid/{missing}", 0),
        ("case.md", f"https://example.invalid/x;{missing} https://example.invalid/x,({missing})", 0),
        ("case.md", f"//example.invalid/x;{missing} https://[::1]/x;{missing}", 0),
        ("case.md", f"**https://example.invalid/({missing})** *_//example.invalid/x;{missing}_*", 0),
        ("case.md", f"old-{missing} vendor/{missing} /opt/{missing}", 0),
        ("case.md", f"{missing}.backup {missing}-backup", 0),
        ("case.md", f"{missing}.md", 1),
        ("case.md", f"./{missing}", 1),
        ("case.md", f"`{missing}`,\n[source]({missing}#L1)\n**{missing}**\n{missing}:1-2.", 4),
        ("case.md", f"***{missing}***\n**_{missing}_**\n**{missing}:1-2.**", 3),
        ("case.md", f"[https://example.invalid]({missing})", 1),
        ("case.md", f"[external](https://example.invalid/a)[source]({missing})", 1),
        ("case.md", f"[external](https://[::1]/a_(b(c)))[source]({missing})", 1),
        ("case.md", f"[source]:{missing}\n [source]: {missing}\n[source]:<{missing}#L1>", 3),
        ("case.md", f"[source]:https://example.invalid/{missing}\n[source]:custom:{missing}", 0),
        ("case.md", f"TODO:{missing}\nhttps://example.invalid/[source]:{missing}", 0),
        ("case.md", "cuda-oxide/crates/../scripts/no-such-source-reference-file.sh", 1),
        ("case.rs", f"/// {missing}\n//! {missing}", 2),
        ("case.rs", f"///{missing}\n//!{missing}", 2),
        ("case.rs", f'// {missing}\n//// {missing}\nconst P: &str = "{missing}";', 0),
    ]
    for filename, text, expected in controls:
        actual = list(broken_references(references(filename, text), tracked, generated))
        if len(actual) != expected:
            sys.exit(f"error: source-reference guard self-test failed for {text!r}: {actual}")
    traversal = "cuda-oxide/crates/../scripts/no-such-source-reference-file.sh"
    if list(broken_references(references("case.md", traversal), tracked, generated)) != [
        (1, traversal, "contains an unsupported traversal segment")
    ]:
        sys.exit("error: source-reference guard self-test lost the traversal diagnostic")
    if list(paths_in_line(f"`{live}` {output}")) != [live, output]:
        sys.exit("error: source-reference guard self-test failed to extract live/output paths")
    for text in (f'const P: &str = "{output}";', f"/// https://example.invalid/{output}"):
        seen = {path for _, path in references("case.rs", text)}
        if not list(declaration_errors({output}, tracked, seen)):
            sys.exit("error: generated-output self-test accepted a reference outside prose scope")
    if not list(declaration_errors({output}, tracked | {output}, {output})):
        sys.exit("error: generated-output self-test accepted an already tracked output")
    seen = {path for _, path in references("case.md", f"[output]:{output}")}
    if list(declaration_errors({output}, tracked, seen)):
        sys.exit("error: generated-output self-test missed a reference definition")
    nested_script = "cutile-rs/scripts/run_all.sh"
    if nested_script not in tracked:
        sys.exit(f"error: source-reference guard self-test expected {nested_script} to be tracked")
    nested_text = f"`{nested_script[len('cutile-rs/'):]}`"
    if list(broken_references(checked_references("cutile-rs/README.md", nested_text), tracked, generated)):
        sys.exit("error: source-reference guard self-test rejected a cutile-rs scripts reference")
    missing_nested = "scripts/no-such-source-reference-file.sh"
    if list(broken_references(checked_references("cutile-rs/README.md", f"`{missing_nested}`"), tracked, generated)) != [
        (1, "cutile-rs/" + missing_nested, "is not tracked in this repository")
    ]:
        sys.exit("error: source-reference guard self-test missed a broken cutile-rs scripts reference")
    if list(broken_references(checked_references("README.md", f"`scripts/run_all.sh`"), tracked, generated)) != [
        (1, "scripts/run_all.sh", "is not tracked in this repository")
    ]:
        sys.exit("error: source-reference guard self-test accepted an Oxide-root scripts/run_all.sh reference")
    oxide_script = "cuda-oxide/scripts/smoketest.sh"
    if oxide_script not in tracked:
        sys.exit(f"error: source-reference guard self-test expected {oxide_script} to be tracked")
    if list(broken_references(checked_references("cuda-oxide/README.md", "`scripts/smoketest.sh`"), tracked, generated)):
        sys.exit("error: source-reference guard self-test rejected a cuda-oxide scripts reference")
    if list(broken_references(checked_references("cuda-oxide/README.md", "`scripts/run_all.sh`"), tracked, generated)) != [
        (1, "cuda-oxide/scripts/run_all.sh", "is not tracked in this repository")
    ]:
        sys.exit("error: source-reference guard self-test accepted a cutile script as a cuda-oxide script")

checked = 0
broken = []
referenced = set()
for filename in sorted(tracked):
    if pathlib.PurePosixPath(filename).suffix not in (".md", ".rs"):
        continue
    refs = list(checked_references(filename, pathlib.Path(filename).read_text(encoding="utf-8")))
    referenced.update(path for _, path in refs)
    checked += len({path for _, path in refs})
    broken.extend(
        f"{filename}:{lineno}: names {path}, which {reason}"
        for lineno, path, reason in broken_references(refs, tracked, GENERATED_OUTPUT_PATHS)
    )

errors = list(declaration_errors(GENERATED_OUTPUT_PATHS, tracked, referenced))
for error in errors:
    print(f"error: source-reference guard: {error}; update GENERATED_OUTPUT_PATHS", file=sys.stderr)
for problem in broken:
    print(problem, file=sys.stderr)
if errors or broken:
    sys.exit("error: source references failed; repoint stale prose and declare generated outputs explicitly")
print(f"source references ok: {checked} repo-anchored paths in prose, all tracked or declared generated outputs")
PY
