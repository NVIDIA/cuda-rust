#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Verify every copy of the CUDA toolkit discovery rules still agrees with
# cuda-bindings/build.rs.
#
# When neither CUDA_TOOLKIT_PATH nor CUDA_HOME is set, cuda-bindings walks a
# fixed list of install roots and takes the first one whose cuda.h clears the
# CUDA 13.0 floor.  Build scripts cannot import each other and doctor cannot
# import a build script, so that contract -- the variable order, the floor and
# the candidate list for each cfg arm -- is copied by hand into:
#
#   1. cuda-core/build.rs (TOOLKIT_ENV_VARS, MIN_CUDA_VERSION,
#      DEFAULT_TOOLKIT_DIRS).  It probes the same cuda.h for the multicast
#      API.  When it resolves a different toolkit, or none, the
#      cuda_has_multicast cfg goes quiet: the multicast types vanish from
#      cuda-core and multicast_supported() answers Ok(false) on every device.
#      The build still succeeds, so nothing reports it.
#
#   2. cuda-oxide/crates/cargo-oxide/src/commands/doctor.rs (TOOLKIT_ENV_VARS,
#      MIN_CUDA_VERSION, DEFAULT_TOOLKIT_CANDIDATES).  Doctor exists to explain
#      what the build resolved, so a stale copy reports the wrong toolkit, or
#      fails a toolkit the build accepts.
#
#   3. The search order as documented in cuda-bindings/README.md and in the
#      book's installation page (getting-started/installation.md).  Readers
#      decide where to install, or whether to set a variable at all, from it.
#
# Both code copies have drifted before.  Doctor's once answered
# /usr/local/cuda for an unset environment while the build walked the list
# (#1265), and /opt/cuda, where Arch Linux and its derivatives install the
# toolkit, was in cuda-bindings but in neither copy nor either document.
#
# The documents are checked for the non-Windows list only; the README spells
# the Windows roots in shortened prose ("v13.3 and v13.2").
#
# Run this after changing toolkit discovery in cuda-bindings/build.rs.
set -euo pipefail

# The comparisons below are byte-wise; pin the locale to match.
export LC_ALL=C

cd "$(dirname "$0")/../.."

BINDINGS=cuda-bindings/build.rs
CORE=cuda-core/build.rs
DOCTOR=cuda-oxide/crates/cargo-oxide/src/commands/doctor.rs
README=cuda-bindings/README.md
BOOK=cuda-oxide/cuda-oxide-book/getting-started/installation.md

if ! command -v python3 >/dev/null 2>&1; then
    echo "error: python3 is required to verify toolkit discovery parity" >&2
    echo "       refusing to report success from a check that cannot run" >&2
    exit 1
fi

for path in "${BINDINGS}" "${CORE}" "${DOCTOR}" "${README}" "${BOOK}"; do
    test -s "${path}"
done

python3 - "${BINDINGS}" "${CORE}" "${DOCTOR}" "${README}" "${BOOK}" <<'PY'
import re
import sys

bindings_path, core_path, doctor_path, readme_path, book_path = sys.argv[1:6]

ARMS = ("not(windows)", "windows")

# A Rust string literal, plain or raw.  The Windows roots are raw strings
# full of backslashes, so the contents are taken verbatim, never unescaped.
STRING = re.compile(r'r?"([^"]*)"')
BACKTICKED = re.compile(r"`([^`]+)`")


def read(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def strings(span):
    return STRING.findall(span)


def array_arms(text, name):
    """{arm: [entries]} for each `#[cfg(<arm>)]` binding of `name`.

    Matches `const` and `let` bindings with any visibility and any type
    annotation, so reshaping a declaration does not hide it from this check.
    """
    arms = {}
    pattern = re.compile(
        r"#\[cfg\((not\(windows\)|windows)\)\]\s*"
        r"(?:pub(?:\([^)]*\))?\s+)?(?:const|let)\s+" + re.escape(name) +
        r"\b[^=]*=\s*&?\[(.*?)\];",
        re.S,
    )
    for arm, body in pattern.findall(text):
        if arm in arms:
            sys.exit(f"parse self-test failed: two #[cfg({arm})] bindings of {name}")
        arms[arm] = strings(body)
    return arms


def string_array(text, name):
    match = re.search(
        r"const\s+" + re.escape(name) + r"\b[^=]*=\s*&?\[(.*?)\];", text, re.S
    )
    return strings(match.group(1)) if match else None


def number(text, name):
    match = re.search(r"const\s+" + re.escape(name) + r"\b[^=]*=\s*([0-9_]+)\s*;", text)
    return int(match.group(1).replace("_", "")) if match else None


def contract(path, candidates_name):
    text = read(path)
    return {
        "TOOLKIT_ENV_VARS": string_array(text, "TOOLKIT_ENV_VARS"),
        "MIN_CUDA_VERSION": number(text, "MIN_CUDA_VERSION"),
        candidates_name: array_arms(text, candidates_name),
    }


reference = contract(bindings_path, "candidates")
reference_arms = reference.pop("candidates")

# Parse self-test.  A guard whose failure mode is "matched nothing" has to
# prove it still reads its reference before a clean result means anything.
if (
    not reference["TOOLKIT_ENV_VARS"]
    or not reference["MIN_CUDA_VERSION"]
    or any(not reference_arms.get(arm) for arm in ARMS)
):
    sys.exit(
        f"parse self-test failed: read env vars={reference['TOOLKIT_ENV_VARS']!r}, "
        f"floor={reference['MIN_CUDA_VERSION']!r}, candidates={reference_arms!r} "
        f"from {bindings_path}; the file layout changed, fix this script before "
        "trusting it"
    )

failures = []


def compare(path, what, found, expected):
    if found is None:
        failures.append(f"{path}: {what} not found; expected {expected!r}")
    elif found != expected:
        failures.append(f"{path}: {what} is {found!r}; {bindings_path} has {expected!r}")


for path, candidates_name in (
    (core_path, "DEFAULT_TOOLKIT_DIRS"),
    (doctor_path, "DEFAULT_TOOLKIT_CANDIDATES"),
):
    mirror = contract(path, candidates_name)
    for key in ("TOOLKIT_ENV_VARS", "MIN_CUDA_VERSION"):
        compare(path, key, mirror[key], reference[key])
    for arm in ARMS:
        compare(
            path,
            f"#[cfg({arm})] {candidates_name}",
            mirror[candidates_name].get(arm),
            reference_arms[arm],
        )

unix_roots = reference_arms["not(windows)"]

# README: "... the standard install locations (on Linux `a`, `b`; on Windows
# ...)".  Every backticked path between "on Linux" and the semicolon.
readme = re.search(r"on Linux(.*?);\s*on Windows", read(readme_path), re.S)
compare(
    readme_path,
    "the documented Linux search order",
    BACKTICKED.findall(readme.group(1)) if readme else None,
    unix_roots,
)

# Book: "... the first one holding a `cuda.h` of CUDA 13.0 or newer wins: `a`,
# `b`, `c`."  Everything backticked from "wins:" to the end of that sentence.
book = re.search(r"\bwins:\s*(.*?)\.(?:\s|$)", read(book_path), re.S)
compare(
    book_path,
    "the documented search order",
    BACKTICKED.findall(book.group(1)) if book else None,
    unix_roots,
)

if failures:
    print("CUDA toolkit discovery copies disagree with cuda-bindings/build.rs:", file=sys.stderr)
    for failure in failures:
        print(f"  {failure}", file=sys.stderr)
    print(
        "Update the copies above to match, or update cuda-bindings/build.rs if it is the one that is wrong.",
        file=sys.stderr,
    )
    sys.exit(1)

print(
    f"OK: {len(unix_roots)} default roots ({len(reference_arms['windows'])} on Windows), "
    f"{len(reference['TOOLKIT_ENV_VARS'])} variables and the "
    f"{reference['MIN_CUDA_VERSION']} floor agree across cuda-core, doctor and both documents."
)
PY
