#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Run with a CUDA Toolkit and GPU matching the requested architecture.
set -euo pipefail

cd "$(dirname "$0")/.."
arch="${1:-sm_120}"
example="$PWD/crates/rustc-codegen-cuda/examples/external_ltoir"
artifacts="$(mktemp -d "${TMPDIR:-/tmp}/oxide-external-ltoir.XXXXXX")"
echo "External LTO-IR regression artifacts: $artifacts"
# Keep the example's executable location independent of the caller's Cargo cache.
unset CARGO_TARGET_DIR
oxide=(cargo oxide)
if [[ -n "${CARGO_OXIDE_BIN:-}" ]]; then
    oxide=("$CARGO_OXIDE_BIN")
fi

nvcc_flags=(--ltoir -dlto -arch="$arch")
if [[ -n "${NVCC_CCBIN:-${CUDAHOSTCXX:-}}" ]]; then
    nvcc_flags+=(-ccbin="${NVCC_CCBIN:-$CUDAHOSTCXX}")
fi
nvcc "${nvcc_flags[@]}" "$example/scale.cu" -o "$artifacts/scale.ltoir"
nvcc "${nvcc_flags[@]}" -DEXTERNAL_ADDEND=3 "$example/add.cu" -o "$artifacts/add.ltoir"

flags=(--arch "$arch" --materialize-cubin
    --link-ltoir "$artifacts/scale.ltoir" --link-ltoir "$artifacts/add.ltoir")
cargo_args=(--manifest-path "$example/Cargo.toml" --release)

run() {
    local label="$1"
    shift
    if ! "$@" >"$artifacts/$label.log" 2>&1; then
        cat "$artifacts/$label.log" >&2
        return 1
    fi
    echo "PASS $label"
}

run run "${oxide[@]}" run external_ltoir "${flags[@]}"
run test "${oxide[@]}" test "${flags[@]}" -- "${cargo_args[@]}" -- --nocapture
run build "${oxide[@]}" build "${flags[@]}" -- "${cargo_args[@]}" -v
run fresh "${oxide[@]}" build "${flags[@]}" -- "${cargo_args[@]}" -v
grep -q 'Fresh external_ltoir ' "$artifacts/fresh.log"

# Change just the library content, retaining its path, size and modification
# time. The next passthrough build must invalidate Cargo's cached device code.
cp -p "$artifacts/add.ltoir" "$artifacts/add-original.ltoir"
nvcc "${nvcc_flags[@]}" -DEXTERNAL_ADDEND=1 "$example/add.cu" -o "$artifacts/add.ltoir"
if [[ "$(wc -c < "$artifacts/add-original.ltoir")" != "$(wc -c < "$artifacts/add.ltoir")" ]]; then
    echo "error: this nvcc did not produce equal-sized addend variants" >&2
    exit 1
fi
if cmp -s "$artifacts/add-original.ltoir" "$artifacts/add.ltoir"; then
    echo "error: the addend variants have identical bytes" >&2
    exit 1
fi
touch -r "$artifacts/add-original.ltoir" "$artifacts/add.ltoir"
run content-change "${oxide[@]}" build "${flags[@]}" -- "${cargo_args[@]}" -v
mv "$artifacts/scale.ltoir" "$artifacts/scale.saved"
mv "$artifacts/add.ltoir" "$artifacts/add.saved"
run changed-result "$example/target/release/external_ltoir" 1
mv "$artifacts/scale.saved" "$artifacts/scale.ltoir"
mv "$artifacts/add.saved" "$artifacts/add.ltoir"

if "${oxide[@]}" build --arch "$arch" --materialize-cubin \
    --link-ltoir "$artifacts/scale.ltoir" -- "${cargo_args[@]}" \
    >"$artifacts/missing-symbol.log" 2>&1; then
    echo "error: a missing transitive device definition unexpectedly linked" >&2
    exit 1
fi
grep -q 'external_add' "$artifacts/missing-symbol.log"
grep -qi 'nvJitLink' "$artifacts/missing-symbol.log"
echo "PASS missing-symbol"

printf 'not an LTO-IR module\n' >"$artifacts/corrupt.ltoir"
if "${oxide[@]}" build --arch "$arch" --materialize-cubin \
    --link-ltoir "$artifacts/scale.ltoir" --link-ltoir "$artifacts/corrupt.ltoir" \
    -- "${cargo_args[@]}" >"$artifacts/corrupt-input.log" 2>&1; then
    echo "error: corrupt LTO-IR unexpectedly linked" >&2
    exit 1
fi
grep -qi 'nvJitLink' "$artifacts/corrupt-input.log"
grep -q 'corrupt.ltoir' "$artifacts/corrupt-input.log"
echo "PASS corrupt-input"

# Exercise the backend guard using a disposable output directory. The supplied
# input and a previous output must both survive rejection before codegen writes.
collision="$artifacts/collision"
mkdir -p "$collision"
cp "$artifacts/scale.ltoir" "$collision/external_ltoir.ll"
printf 'previous output sentinel\n' >"$artifacts/previous-output"
cp "$artifacts/previous-output" "$collision/external_ltoir.ptx"
if CUDA_OXIDE_PTX_DIR="$collision" "${oxide[@]}" build --arch "$arch" --materialize-cubin \
    --link-ltoir "$collision/external_ltoir.ll" --link-ltoir "$artifacts/add.ltoir" \
    -- "${cargo_args[@]}" >"$artifacts/input-output-overlap.log" 2>&1; then
    echo "error: an external input overlapping generated output was accepted" >&2
    exit 1
fi
grep -q 'overlaps generated artifact' "$artifacts/input-output-overlap.log"
cmp "$artifacts/scale.ltoir" "$collision/external_ltoir.ll"
cmp "$artifacts/previous-output" "$collision/external_ltoir.ptx"
echo "PASS input-output-overlap"

cp -p "$artifacts/add-original.ltoir" "$artifacts/add.ltoir"
run restored "${oxide[@]}" run external_ltoir "${flags[@]}"
