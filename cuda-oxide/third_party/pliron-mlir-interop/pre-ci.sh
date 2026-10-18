#!/bin/bash

# Pre-CI tests to run locally before pushing to GitHub.
#
# Set $MLIR_OPT to verify translated MLIR.

set -ex

cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
RUSTFLAGS="-D warnings" cargo test --workspace
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --document-private-items
