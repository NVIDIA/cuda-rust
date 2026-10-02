#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build the umbrella landing page, then each component book, and stitch the
# output into one site:
#
#   build/html/                 umbrella landing page
#   build/html/cuda-oxide/      SIMT model book
#   build/html/cutile/          Tile model book
#
# Modelled on cuda-python's cuda_python/docs/build_all_docs.sh.

set -ex

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

SPHINX_OPTS=${SPHINXOPTS:-"-W --keep-going"}

# Component book locations, relative to the repository root. Override to build
# against checkouts that live elsewhere.
REPO_ROOT=${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}
OXIDE_BOOK=${OXIDE_BOOK:-"${REPO_ROOT}/cuda-oxide-book"}
CUTILE_BOOK=${CUTILE_BOOK:-"${REPO_ROOT}/crates/cutile-rs/cutile-book"}

OUT="${SCRIPT_DIR}/build/html"
rm -rf "${SCRIPT_DIR}/build"

# Umbrella landing page at the site root.
sphinx-build ${SPHINX_OPTS} -b html "${SCRIPT_DIR}" "${OUT}"

build_component() {
    local name=$1 src=$2
    if [[ ! -d "${src}" ]]; then
        echo "skip: ${name} book not found at ${src}" >&2
        return 0
    fi
    local dest="${OUT}/${name}"
    mkdir -p "${dest}"
    sphinx-build ${SPHINX_OPTS} -b html "${src}" "${dest}"
}

build_component cuda-oxide "${OXIDE_BOOK}"
build_component cutile     "${CUTILE_BOOK}"
