#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

root="$(cd "$(dirname "$0")" && pwd)"
ptx="${1:-$root/copy_aggregate_borrow.ptx}"

if [[ ! -f "$ptx" ]]; then
  echo "PTX file not found: $ptx" >&2
  exit 1
fi

body="$(awk '
  /\.visible[[:space:]]+\.entry[[:space:]]+borrowed_copy_aggregate\(/ {
    inside = 1
  }
  inside {
    print
    opens += gsub(/\{/, "{")
    closes += gsub(/\}/, "}")
    if (opens > 0 && opens == closes) {
      exit
    }
  }
' "$ptx")"

if [[ -z "$body" ]]; then
  echo "entry not found: borrowed_copy_aggregate" >&2
  exit 1
fi

if grep -Eq '(^|[[:space:]])\.local|ld\.local|st\.local' <<<"$body"; then
  echo "unexpected local-memory operation in borrowed_copy_aggregate" >&2
  printf '%s\n' "$body" >&2
  exit 1
fi

echo "copy_aggregate_borrow PTX shape: PASS"
