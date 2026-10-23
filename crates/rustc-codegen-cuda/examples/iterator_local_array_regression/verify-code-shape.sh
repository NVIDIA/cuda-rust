#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# #399: a four-element local array consumed through runtime-bound iterator
# adapters must not reappear as PTX local storage in the generated kernel.
#
# Usage:
#   verify-code-shape.sh [PTX_PATH]
#
# With no argument the script exercises itself against the fixtures below and
# then checks the example's generated PTX, which is the form the repository
# smoke test calls. With an explicit path it checks only that file, with no
# self-test, so a caller can point it at a specific build's output and get a
# predictable answer.
#
# `indexed_array_control` is a semantic comparison, not a performance contract:
# it may or may not use local memory and its state is reported rather than
# asserted. Gating on it would fail this script the day that kernel is
# optimized too.
set -euo pipefail

SELF_TEST=0
if [[ $# -eq 0 ]]; then
  root="$(cd "$(dirname "$0")" && pwd)"
  ptx="$root/iterator_local_array_regression.ptx"
  SELF_TEST=1
else
  ptx="$1"
fi

# Print the body of `.visible .entry $1` in $2, or fail when the entry is
# absent, truncated, or its braces do not balance.
entry_body() {
  awk -v entry="$1" '
    $0 ~ "^\\.visible[[:space:]]+\\.entry[[:space:]]+" entry "\\(" { inside = 1 }
    inside {
      print
      opens += gsub(/\{/, "{")
      closes += gsub(/\}/, "}")
      if (opens > 0 && opens == closes) { complete = 1; exit }
    }
    END { if (!complete) exit 1 }
  ' "$2"
}

has_local() {
  grep -Eq '(^|[[:space:]])\.local|ld\.local|st\.local' <<<"$1"
}

check() {
  local file="$1"

  if [[ ! -f "$file" ]]; then
    echo "PTX file not found: $file" >&2
    return 1
  fi
  if [[ ! -s "$file" ]]; then
    echo "PTX file is empty: $file" >&2
    return 1
  fi

  local iterator_body
  if ! iterator_body="$(entry_body iterator_consumed_array "$file")"; then
    echo "iterator_consumed_array entry not found, truncated, or unbalanced in $file" >&2
    return 1
  fi

  # Only the target kernel's body is inspected: local memory in an unrelated
  # kernel says nothing about this one.
  if has_local "$iterator_body"; then
    echo "unexpected local-memory operation in iterator_consumed_array" >&2
    printf '%s\n' "$iterator_body" >&2
    return 1
  fi

  local control_body control_state
  if control_body="$(entry_body indexed_array_control "$file")"; then
    if has_local "$control_body"; then
      control_state="uses local memory"
    else
      control_state="scalarized too"
    fi
  else
    control_state="not present"
  fi

  echo "iterator_local_array_regression PTX shape: PASS (indexed_array_control: $control_state)"
}

# Minimal hand-written fragments for the parser and the local-memory rule, not
# captured compiler output.
self_test() {
  local dir status=0
  dir="$(mktemp -d)"
  trap 'rm -rf "$dir"' RETURN

  local clean_body local_body
  clean_body='  .reg .b32 %r<2>;
  mov.u32 %r1, 0;'
  local_body='  .local .align 4 .b8 __local_depot0[16];
  st.local.b32 [%r1], 0;'

  expect() {
    local want="$1" name="$2" file="$3" got
    if check "$file" >/dev/null 2>&1; then got=0; else got=1; fi
    if [[ "$got" != "$want" ]]; then
      echo "self-test fixture '$name': expected exit $want, got $got" >&2
      status=1
    fi
  }

  write() {
    local file="$1" iterator="$2" control="$3"
    {
      printf '.visible .entry iterator_consumed_array(\n{\n%s\n}\n' "$iterator"
      if [[ -n "$control" ]]; then
        printf '.visible .entry indexed_array_control(\n{\n%s\n}\n' "$control"
      fi
    } > "$file"
  }

  write "$dir/clean-with-local-control.ptx" "$clean_body" "$local_body"
  write "$dir/clean-everywhere.ptx" "$clean_body" "$clean_body"
  write "$dir/iterator-declares-local.ptx" "$local_body" "$local_body"
  write "$dir/iterator-loads-local.ptx" '  ld.local.b32 %r1, [%r1];' "$clean_body"
  write "$dir/control-absent.ptx" "$clean_body" ""
  printf '.visible .entry iterator_consumed_array_extra(\n{\n%s\n}\n' "$clean_body" > "$dir/prefix-only.ptx"
  printf '// iterator_consumed_array in a comment\n.visible .entry other(\n{\n%s\n}\n' "$clean_body" > "$dir/comment-only.ptx"
  printf '.visible .entry iterator_consumed_array(\n{\n%s\n' "$clean_body" > "$dir/truncated.ptx"
  printf '' > "$dir/empty.ptx"
  {
    printf '.visible .entry iterator_consumed_array(\n{\n%s\n}\n' "$clean_body"
    printf '.visible .entry indexed_array_control(\n{\n%s\n}\n' "$clean_body"
    printf '.visible .entry unrelated(\n{\n%s\n}\n' "$local_body"
  } > "$dir/unrelated-kernel-has-local.ptx"

  expect 0 "iterator clean, control uses local" "$dir/clean-with-local-control.ptx"
  expect 0 "iterator clean, control scalarized too" "$dir/clean-everywhere.ptx"
  expect 1 "iterator declares .local" "$dir/iterator-declares-local.ptx"
  expect 1 "iterator loads local" "$dir/iterator-loads-local.ptx"
  expect 0 "control absent" "$dir/control-absent.ptx"
  expect 1 "entry name only as a prefix" "$dir/prefix-only.ptx"
  expect 1 "entry named only in a comment" "$dir/comment-only.ptx"
  expect 1 "entry body truncated" "$dir/truncated.ptx"
  expect 1 "empty file" "$dir/empty.ptx"
  expect 0 "unrelated kernel uses local" "$dir/unrelated-kernel-has-local.ptx"

  return $status
}

if [[ "$SELF_TEST" == 1 ]]; then
  if ! self_test; then
    echo "verify-code-shape.sh self-test: FAIL" >&2
    exit 1
  fi
fi

check "$ptx"
