#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SPHINX_BUILD="${SPHINX_BUILD:-sphinx-build}"
OUT_DIR="${CUTILE_DOCS_SITE_DIR:-$REPO_ROOT/_site}"
MAIN_REF="${CUTILE_DOCS_MAIN_REF:-HEAD}"
MAIN_VERSION="${CUTILE_DOCS_MAIN_VERSION:-main}"
TAG_PATTERN="${CUTILE_DOCS_TAG_PATTERN:-cutile-rs/v*}"
BASE_URL="${CUTILE_DOCS_BASE_URL:-/cuda-oxide/cutile-rs/}"

if [[ "$BASE_URL" != /* ]]; then
    BASE_URL="/$BASE_URL"
fi
if [[ "$BASE_URL" != */ ]]; then
    BASE_URL="$BASE_URL/"
fi

TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/cutile-docs.XXXXXX")"
WORKTREES=()

cleanup() {
    for worktree in "${WORKTREES[@]}"; do
        git -C "$REPO_ROOT" worktree remove --force "$worktree" >/dev/null 2>&1 || true
    done
    rm -rf "$TMP_ROOT"
}
trap cleanup EXIT

if [[ -n "${CUTILE_DOCS_TAGS+x}" ]]; then
    # Space- or newline-separated explicit tag list. Set to an empty string to
    # build only the main docs.
    if [[ -z "$CUTILE_DOCS_TAGS" ]]; then
        TAGS=()
    else
        readarray -t TAGS < <(printf '%s\n' $CUTILE_DOCS_TAGS)
    fi
else
    readarray -t TAGS < <(git -C "$REPO_ROOT" tag --list "$TAG_PATTERN" --sort=-v:refname)
fi

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR/_static"
touch "$OUT_DIR/.nojekyll"

write_versions_json() {
    python3 - "$OUT_DIR/_static/versions.json" "$BASE_URL" "$MAIN_VERSION" "${TAGS[@]}" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
base_url = sys.argv[2]
main_version = sys.argv[3]
tags = sys.argv[4:]

def version_url(version: str) -> str:
    return f"{base_url}{version}/"

def display_version(ref: str) -> str:
    # Monorepo tags are cutile-rs/v0.2.0 → directory/switcher name 0.2.0.
    prefix = "cutile-rs/"
    if not ref.startswith(prefix):
        raise SystemExit(f"expected cutile-rs/ namespaced tag, got {ref!r}")
    version = ref[len(prefix):]
    if not version.startswith("v"):
        raise SystemExit(f"expected leading v after {prefix}, got {ref!r}")
    return version[1:]

versions = [
    {
        "name": main_version,
        "version": main_version,
        "url": version_url(main_version),
    }
]

for tag in tags:
    version = display_version(tag)
    entry = {
        "name": version,
        "version": version,
        "url": version_url(version),
    }
    versions.append(entry)

output.write_text(json.dumps(versions, indent=2) + "\n", encoding="utf-8")
PY
}

write_root_index() {
    cat > "$OUT_DIR/index.html" <<EOF
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta http-equiv="refresh" content="0; url=${MAIN_VERSION}/">
    <title>cuTile Rust Documentation</title>
  </head>
  <body>
    <p>Redirecting to <a href="${MAIN_VERSION}/">cuTile Rust ${MAIN_VERSION} documentation</a>.</p>
  </body>
</html>
EOF
}

build_ref() {
    local ref="$1"
    local version="$2"
    local src="$REPO_ROOT"
    local out="$OUT_DIR/$version"

    if [[ "$ref" != "HEAD" ]]; then
        src="$TMP_ROOT/$version"
        git -C "$REPO_ROOT" worktree add --detach "$src" "$ref" >/dev/null
        WORKTREES+=("$src")
    fi

    local book_src=""
    if [[ -f "$src/cutile-book/conf.py" ]]; then
        book_src="$src/cutile-book"
    elif [[ -f "$src/cutile-rs/cutile-book/conf.py" ]]; then
        # Namespaced tags were rewritten under cutile-rs/ by filter-repo.
        book_src="$src/cutile-rs/cutile-book"
    else
        echo "Skipping $version: cutile-book/conf.py not found at $ref" >&2
        return
    fi

    echo "Building cuTile book for $version ($ref)"
    CUTILE_DOCS_VERSION="$version" \
    CUTILE_DOCS_SWITCHER_JSON="${BASE_URL}_static/versions.json" \
        "$SPHINX_BUILD" -b html "$book_src" "$out"

    # Older tags may enable sphinx-sitemap with unversioned URLs. The versioned
    # Pages layout does not need per-version sitemap files.
    rm -f "$out/sitemap.xml" "$out/sitemap.xml.gz"
}

write_versions_json
write_root_index

build_ref "$MAIN_REF" "$MAIN_VERSION"

for tag in "${TAGS[@]}"; do
    [[ -n "$tag" ]] || continue
    # cutile-rs/v0.2.0 → 0.2.0 (must match display_version() above).
    if [[ "$tag" != cutile-rs/v* ]]; then
        echo "expected cutile-rs/v* tag, got: $tag" >&2
        exit 1
    fi
    version="${tag#cutile-rs/v}"
    build_ref "$tag" "$version"
done

echo "Built versioned documentation into $OUT_DIR"
