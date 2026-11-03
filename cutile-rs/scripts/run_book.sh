#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BOOK_DIR="$REPO_ROOT/cutile-book"
VENV_DIR="${CUTILE_BOOK_VENV:-$BOOK_DIR/.venv}"
HOST="${CUTILE_BOOK_HOST:-127.0.0.1}"
PORT="${CUTILE_BOOK_PORT:-8000}"
OPEN_BROWSER="${CUTILE_BOOK_OPEN_BROWSER:-1}"

usage() {
    cat <<EOF
Usage: scripts/run_book.sh [serve|build|setup|clean|help]

Commands:
  serve   Build and serve the book with live reload (default)
  build   Build static HTML into cutile-book/_build/html
  setup   Create/update cutile-book/.venv and install requirements
  clean   Remove cutile-book/_build
  help    Show this help

Environment:
  CUTILE_BOOK_HOST          Host for the live server (default: 127.0.0.1)
  CUTILE_BOOK_PORT          Port for the live server (default: 8000)
  CUTILE_BOOK_OPEN_BROWSER  Set to 0 to avoid opening a browser (default: 1)
  CUTILE_BOOK_VENV          Override the virtualenv path
EOF
}

setup_venv() {
    if [[ ! -x "$VENV_DIR/bin/python" ]]; then
        python3 -m venv "$VENV_DIR"
    fi

    "$VENV_DIR/bin/pip" install --upgrade pip
    "$VENV_DIR/bin/pip" install -r "$BOOK_DIR/requirements.txt"
}

ensure_venv() {
    if [[ ! -x "$VENV_DIR/bin/sphinx-build" || ! -x "$VENV_DIR/bin/sphinx-autobuild" ]]; then
        setup_venv
    fi
}

build_html() {
    ensure_venv
    cd "$BOOK_DIR"
    "$VENV_DIR/bin/sphinx-build" -b html . _build/html
    echo "Built $BOOK_DIR/_build/html/index.html"
}

serve_html() {
    ensure_venv
    cd "$BOOK_DIR"

    open_arg=()
    if [[ "$OPEN_BROWSER" != "0" ]]; then
        open_arg=(--open-browser)
    fi

    echo "Serving cuTile book at http://$HOST:$PORT/"
    echo "Press Ctrl-C to stop."
    "$VENV_DIR/bin/sphinx-autobuild" . _build/html \
        --host "$HOST" \
        --port "$PORT" \
        "${open_arg[@]}" \
        --write-all \
        --watch . \
        --ignore "*.pyc" \
        --ignore ".venv/*" \
        --ignore "_build/*" \
        --ignore "book/*"
}

command="${1:-serve}"

case "$command" in
    serve|run|livehtml)
        serve_html
        ;;
    build|html)
        build_html
        ;;
    setup)
        setup_venv
        ;;
    clean)
        rm -rf "$BOOK_DIR/_build"
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
