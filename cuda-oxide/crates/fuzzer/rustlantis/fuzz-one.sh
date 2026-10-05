#!/bin/sh

echo "$1"

TMPDIR="${TMPDIR:-/tmp}"

target/release/generate $1 | target/release/difftest -
if [ $? -ne 0 ]; then
    SOURCE_DEBUG="$TMPDIR/$1-debug.rs"
    target/release/generate --debug $1 > $SOURCE_DEBUG

    REPRO_DIR="${REPRO_DIR:-repros/}"
    mkdir -p $REPRO_DIR

    if target/release/difftest $SOURCE_DEBUG 2> /dev/null; then
        target/release/generate $1 > "$REPRO_DIR/$1.rs"
    else
        mv $SOURCE_DEBUG $REPRO_DIR
    fi
    exit 1
fi
