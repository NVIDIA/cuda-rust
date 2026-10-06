#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build the umbrella landing page, then each component book, and stitch the
# output into one site:
#
#   build/html/                 umbrella landing page
#   build/html/cuda-oxide/      SIMT model book
#   build/html/cutile/          Versioned Tile model book
#
# Modelled on cuda-python's cuda_python/docs/build_all_docs.sh.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

SPHINX_OPTS=${SPHINXOPTS:--W --keep-going}
read -r -a SPHINX_FLAGS <<< "${SPHINX_OPTS}"

# Component book locations, relative to the repository root. Override to build
# against checkouts that live elsewhere.
REPO_ROOT=${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}
OXIDE_BOOK=${OXIDE_BOOK:-"${REPO_ROOT}/cuda-oxide/cuda-oxide-book"}
CUTILE_BOOK="${REPO_ROOT}/cutile-rs/cutile-book"
CUTILE_BUILDER="${REPO_ROOT}/cutile-rs/scripts/build_versioned_book.sh"

for book in "${OXIDE_BOOK}" "${CUTILE_BOOK}"; do
    if [[ ! -f "${book}/conf.py" ]]; then
        echo "missing component book: ${book}" >&2
        exit 1
    fi
done
if [[ ! -f "${CUTILE_BUILDER}" ]]; then
    echo "missing Tile book builder: ${CUTILE_BUILDER}" >&2
    exit 1
fi

OUT="${SCRIPT_DIR}/build/html"
rm -rf "${SCRIPT_DIR}/build"

# Umbrella landing page at the site root.
sphinx-build "${SPHINX_FLAGS[@]}" -b html "${SCRIPT_DIR}" "${OUT}"
touch "${OUT}/.nojekyll"

# SIMT book uses the same warning policy as the landing page.
sphinx-build "${SPHINX_FLAGS[@]}" -b html "${OXIDE_BOOK}" "${OUT}/cuda-oxide"

# Preserve the Tile book's version switcher, latest redirect, and historical
# tags. The versioned builder owns everything below build/html/cutile/.
SPHINXOPTS="${SPHINX_OPTS}" \
CUTILE_DOCS_SITE_DIR="${OUT}/cutile" \
CUTILE_DOCS_BASE_URL="/cuda-rust/cutile/" \
CUTILE_DOCS_MAIN_VERSION="latest" \
    bash "${CUTILE_BUILDER}"
