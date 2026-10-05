#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

TUNING="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP1="$(cd "$TUNING/.." && pwd)"
GEMM_DIR="$EXP1/rust/gemm"
ELEM_DIR="$EXP1/rust/elemwise"
GEMM_TUNING_DIR="$EXP1/tuning/results/gemm"
ELEM_TUNING_DIR="$EXP1/tuning/results/elemwise"

usage() {
  cat <<EOF
usage: $(basename "$0") [--target <all|gemm|elemwise>]
       $(basename "$0") [all|gemm|elemwise]

Runs the two-stage Rust tile + CompileOptions tuning sweep:
  1. Tune tile shape with default optimization hints.
  2. Tune optimization hints at the best tile shape.

Targets:
  all        Run GEMM and elementwise sweeps (default)
  gemm       Run only benchmarks/exp1_safety_overhead/rust/gemm
  elemwise   Run only benchmarks/exp1_safety_overhead/rust/elemwise

Environment:
  SM_ARCH    Select architecture-specific default hints (default: sm_120)
  GEMM_VARIANT
             Single GEMM kernel variant: optimized, full-output, safe, or static
             (default: optimized)
  GEMM_VARIANTS
             Space-separated GEMM kernel variants. Overrides GEMM_VARIANT when set.
             Example: GEMM_VARIANTS="full-output optimized static"
  GEMM_TILE_HINTS
             Hint triples used during the GEMM tile sweep
             (default: default,default,default)
  GEMM_HINT_SWEEP
             Set to 0 to skip GEMM stage 2 optimization-hint sweep
             (default: 1)
  GEMM_SIZE  Single GEMM matrix size for backwards compatibility
             (default: 32768)
  GEMM_SIZES Space-separated GEMM matrix sizes. Overrides GEMM_SIZE when set.
  HINTS      Override hint triples for both targets
  GEMM_HINTS Override GEMM hint triples
  ELEM_HINTS Override elementwise hint triples
  MAX_HINT   Skip numeric hint values above this cap (default: 8)
  FAST=1     Use a smaller tile/hint grid
  STRICT=1   Stop on the first failed tuning candidate
EOF
}

TARGET="${TARGET:-all}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      if [[ $# -lt 2 ]]; then
        echo "error: --target requires a value" >&2
        usage >&2
        exit 2
      fi
      TARGET="$2"
      shift 2
      ;;
    --target=*)
      TARGET="${1#--target=}"
      shift
      ;;
    all|both|gemm|elem|elemwise|elementwise|element-wise)
      TARGET="$1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$TARGET" in
  all|both)
    TARGET="all"
    ;;
  gemm)
    TARGET="gemm"
    ;;
  elem|elemwise|elementwise|element-wise)
    TARGET="elemwise"
    ;;
  *)
    echo "error: unknown target: $TARGET" >&2
    usage >&2
    exit 2
    ;;
esac

if [[ -z "${CUDA_TOOLKIT_PATH:-}" ]]; then
  for candidate in /usr/local/cuda-13.2 /usr/local/cuda-13.1 /usr/local/cuda-13 /usr/local/cuda; do
    if [[ -d "$candidate" ]]; then
      export CUDA_TOOLKIT_PATH="$candidate"
      break
    fi
  done
fi

ELEM_OUT="${ELEM_OUT:-$ELEM_TUNING_DIR/elemwise_rust_tune_results.csv}"
SM_ARCH="${SM_ARCH:-sm_120}"
MAX_HINT="${MAX_HINT:-8}"

normalize_gemm_variant() {
  case "$1" in
    optimized|unchecked)
      echo "optimized"
      ;;
    full-output|full_output)
      echo "full-output"
      ;;
    safe|dynamic)
      echo "safe"
      ;;
    static)
      echo "static"
      ;;
    *)
      echo "error: unknown GEMM variant: $1" >&2
      exit 2
      ;;
  esac
}

GEMM_VARIANTS="${GEMM_VARIANTS:-${GEMM_VARIANT:-optimized}}"
normalized_gemm_variants=()
for variant in $GEMM_VARIANTS; do
  normalized_gemm_variants+=("$(normalize_gemm_variant "$variant")")
done
if [[ "${#normalized_gemm_variants[@]}" -eq 0 ]]; then
  echo "error: GEMM_VARIANTS must contain at least one variant" >&2
  exit 2
fi
GEMM_VARIANTS="${normalized_gemm_variants[*]}"

gemm_variant_count="${#normalized_gemm_variants[@]}"
if [[ -z "${GEMM_OUT+x}" ]]; then
  if [[ "$gemm_variant_count" -gt 1 ]]; then
    GEMM_OUT="$GEMM_TUNING_DIR/gemm_rust_multi_variant_tune_results.csv"
  elif [[ "${normalized_gemm_variants[0]}" == "full-output" ]]; then
    GEMM_OUT="$GEMM_TUNING_DIR/gemm_rust_full_output_tune_results.csv"
  else
    GEMM_OUT="$GEMM_TUNING_DIR/gemm_rust_tune_results.csv"
  fi
fi
GEMM_TILE_HINTS="${GEMM_TILE_HINTS:-default,default,default}"
GEMM_HINT_SWEEP="${GEMM_HINT_SWEEP:-1}"

default_gemm_hints() {
  local mode=$1

  case "$SM_ARCH:$mode" in
    sm_100:fast|sm_120:fast)
      echo "default,default,default 1,default,default 1,1,default 2,default,default 4,default,default 2,2,default 2,default,8"
      ;;
    sm_100:full|sm_120:full)
      echo "default,default,default 1,default,default 1,1,default 2,default,default 4,default,default 8,default,default 2,1,default 2,2,default 2,4,default 2,default,4 2,default,8 4,2,default 4,default,8"
      ;;
    *)
      if [[ "$mode" == "fast" ]]; then
        echo "default,default,default 1,default,default 1,1,default 2,default,default 4,default,default 2,2,default 2,default,8"
      else
        echo "default,default,default 1,default,default 1,1,default 2,default,default 4,default,default 8,default,default 2,1,default 2,2,default 2,4,default 2,default,4 2,default,8 4,2,default 4,default,8"
      fi
      ;;
  esac
}

default_elem_hints() {
  local mode=$1

  case "$SM_ARCH:$mode" in
    sm_100:fast|sm_120:fast)
      echo "default,default,default 1,default,default 2,default,default 2,1,default 2,4,default"
      ;;
    sm_100:full|sm_120:full)
      echo "default,default,default 1,default,default 2,default,default 2,1,default 2,2,default 2,4,default 2,default,4 2,default,8"
      ;;
    *)
      if [[ "$mode" == "fast" ]]; then
        echo "default,default,default 1,default,default 2,default,default 2,1,default 2,4,default"
      else
        echo "default,default,default 1,default,default 2,default,default 2,1,default 2,2,default 2,4,default 2,default,4 2,default,8"
      fi
      ;;
  esac
}

if [[ "${FAST:-0}" == "1" ]]; then
  if [[ " $GEMM_VARIANTS " == *" full-output "* ]]; then
    GEMM_TILES="${GEMM_TILES:-128,256,128 256,256,32 256,256,128}"
  else
    GEMM_TILES="${GEMM_TILES:-128,256,64 256,256,64 256,256,128}"
  fi
  ELEM_BS="${ELEM_BS:-1024 2048 4096}"
  GEMM_HINTS="${GEMM_HINTS:-${HINTS:-$(default_gemm_hints fast)}}"
  ELEM_HINTS="${ELEM_HINTS:-${HINTS:-$(default_elem_hints fast)}}"
else
  if [[ " $GEMM_VARIANTS " == *" full-output "* ]]; then
    GEMM_TILES="${GEMM_TILES:-128,256,64 128,256,128 256,256,32 256,256,64 256,256,128}"
  else
    GEMM_TILES="${GEMM_TILES:-128,128,64 128,256,64 128,256,128 256,128,64 256,256,32 256,256,64 256,256,128}"
  fi
  ELEM_BS="${ELEM_BS:-1024 2048 4096 8192 16384}"
  GEMM_HINTS="${GEMM_HINTS:-${HINTS:-$(default_gemm_hints full)}}"
  ELEM_HINTS="${ELEM_HINTS:-${HINTS:-$(default_elem_hints full)}}"
fi

GEMM_SIZE="${GEMM_SIZE:-32768}"
GEMM_SIZES="${GEMM_SIZES:-$GEMM_SIZE}"
ELEM_N="${ELEM_N:-268435456}"
ELEM_VARIANT="${ELEM_VARIANT:-optimized}"

build_crates() {
  if [[ "$TARGET" == "all" || "$TARGET" == "gemm" ]]; then
    cargo build --release --manifest-path "$GEMM_DIR/Cargo.toml"
  fi
  if [[ "$TARGET" == "all" || "$TARGET" == "elemwise" ]]; then
    cargo build --release --manifest-path "$ELEM_DIR/Cargo.toml"
  fi
}

run_candidate() {
  local label=$1
  shift

  if "$@"; then
    return 0
  fi

  local status=$?
  echo "warning: $label failed with exit code $status; skipping candidate" >&2
  if [[ "${STRICT:-0}" == "1" ]]; then
    exit "$status"
  fi
  return 0
}

add_hint_args() {
  local -n out_args=$1
  local cta=$2
  local occupancy=$3
  local max_divisibility=$4

  if [[ "$cta" != "default" ]]; then
    out_args+=(--cta "$cta")
  fi
  if [[ "$occupancy" != "default" ]]; then
    out_args+=(--occupancy "$occupancy")
  fi
  if [[ "$max_divisibility" != "default" ]]; then
    out_args+=(--max-divisibility "$max_divisibility")
  fi
}

hint_component_allowed() {
  local value=$1

  if [[ "$value" == "default" ]]; then
    return 0
  fi
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "error: hint value must be 'default' or a non-negative integer: $value" >&2
    exit 2
  fi
  [[ "$value" -le "$MAX_HINT" ]]
}

hint_allowed() {
  local cta=$1
  local occupancy=$2
  local max_divisibility=$3

  hint_component_allowed "$cta" \
    && hint_component_allowed "$occupancy" \
    && hint_component_allowed "$max_divisibility"
}

is_default_hint() {
  local cta=$1
  local occupancy=$2
  local max_divisibility=$3

  [[ "$cta" == "default" && "$occupancy" == "default" && "$max_divisibility" == "default" ]]
}

run_gemm_candidate() {
  local variant=$1
  local size=$2
  local bm=$3
  local bn=$4
  local bk=$5
  local cta=$6
  local occupancy=$7
  local max_divisibility=$8
  local args=(
    --tune-one
    --size "$size"
    --bm "$bm"
    --bn "$bn"
    --bk "$bk"
    --csv "$GEMM_OUT"
  )

  case "$variant" in
    optimized) ;;
    full-output)
      args+=(--full-output)
      ;;
    safe)
      args+=(--safe)
      ;;
    static)
      args+=(--static)
      ;;
  esac

  add_hint_args args "$cta" "$occupancy" "$max_divisibility"
  echo "gemm variant=$variant size=$size tile=($bm,$bn,$bk) hints=($cta,$occupancy,$max_divisibility)"
  run_candidate \
    "gemm variant=$variant size=$size tile=($bm,$bn,$bk) hints=($cta,$occupancy,$max_divisibility)" \
    "$GEMM_DIR/target/release/gemm_rust" "${args[@]}"
}

run_elem_candidate() {
  local b=$1
  local cta=$2
  local occupancy=$3
  local max_divisibility=$4
  local args=(
    --tune-one
    --variant "$ELEM_VARIANT"
    --n "$ELEM_N"
    --b "$b"
    --csv "$ELEM_OUT"
  )

  add_hint_args args "$cta" "$occupancy" "$max_divisibility"
  echo "elemwise B=$b hints=($cta,$occupancy,$max_divisibility)"
  run_candidate \
    "elemwise B=$b hints=($cta,$occupancy,$max_divisibility)" \
    "$ELEM_DIR/target/release/elemwise_rust" "${args[@]}"
}

best_gemm_tiles_from_csv() {
  awk -F, '
    NR > 1 {
      config = $1
      size = $2
      key = config SUBSEP size
      score = $NF + 0
      if (!(key in seen) || score > best[key]) {
        seen[key] = 1
        best[key] = score
        best_config[key] = config
        best_size[key] = size
        bm[key] = $5
        bn[key] = $6
        bk[key] = $7
      }
    }
    END {
      if (length(seen) == 0) {
        exit 1
      }
      for (key in seen) {
        printf "%s,%s,%s,%s,%s,%.6f\n", best_config[key], best_size[key], bm[key], bn[key], bk[key], best[key]
      }
    }
  ' "$GEMM_OUT"
}

best_elem_b_from_csv() {
  awk -F, '
    NR > 1 {
      score = $NF + 0
      if (!seen || score > best) {
        seen = 1
        best = score
        b = $3
      }
    }
    END {
      if (!seen) {
        exit 1
      }
      printf "%s,%.6f\n", b, best
    }
  ' "$ELEM_OUT"
}

run_gemm_sweep() {
  local bests variant size bm bn bk tflops

  mkdir -p "$(dirname "$GEMM_OUT")"
  rm -f "$GEMM_OUT"
  echo
  echo "GEMM stage 1/2: tile sweep with GEMM_TILE_HINTS"
  for variant in $GEMM_VARIANTS; do
    for size in $GEMM_SIZES; do
      for tile in $GEMM_TILES; do
        IFS=, read -r bm bn bk <<< "$tile"
        for hint in $GEMM_TILE_HINTS; do
          IFS=, read -r cta occupancy max_divisibility <<< "$hint"
          if ! hint_allowed "$cta" "$occupancy" "$max_divisibility"; then
            echo "skip gemm variant=$variant size=$size tile=($bm,$bn,$bk) hints=($cta,$occupancy,$max_divisibility): exceeds MAX_HINT=$MAX_HINT"
            continue
          fi
          run_gemm_candidate "$variant" "$size" "$bm" "$bn" "$bk" "$cta" "$occupancy" "$max_divisibility"
        done
      done
    done
  done

  if ! bests="$(best_gemm_tiles_from_csv)"; then
    echo "error: GEMM tile sweep produced no results" >&2
    return 1
  fi
  echo "GEMM best tile after stage 1 by variant and size:"
  while IFS=, read -r config size bm bn bk tflops; do
    echo "  variant=$config size=$size tile=($bm,$bn,$bk) at ${tflops} TFLOP/s"
  done <<< "$bests"

  if [[ "$GEMM_HINT_SWEEP" == "0" ]]; then
    echo "GEMM stage 2/2: skipped because GEMM_HINT_SWEEP=0"
    return 0
  fi

  echo
  echo "GEMM stage 2/2: optimization-hint sweep at each variant/size best tile"
  while IFS=, read -r config size bm bn bk tflops; do
    variant="$(normalize_gemm_variant "$config")"
    for hint in $GEMM_HINTS; do
      IFS=, read -r cta occupancy max_divisibility <<< "$hint"
      if is_default_hint "$cta" "$occupancy" "$max_divisibility"; then
        continue
      fi
      if ! hint_allowed "$cta" "$occupancy" "$max_divisibility"; then
        echo "skip gemm variant=$variant size=$size tile=($bm,$bn,$bk) hints=($cta,$occupancy,$max_divisibility): exceeds MAX_HINT=$MAX_HINT"
        continue
      fi
      run_gemm_candidate "$variant" "$size" "$bm" "$bn" "$bk" "$cta" "$occupancy" "$max_divisibility"
    done
  done <<< "$bests"
}

run_elem_sweep() {
  local best b gbps

  mkdir -p "$(dirname "$ELEM_OUT")"
  rm -f "$ELEM_OUT"
  echo
  echo "Elementwise stage 1/2: tile sweep with default optimization hints"
  for b in $ELEM_BS; do
    run_elem_candidate "$b" default default default
  done

  if ! best="$(best_elem_b_from_csv)"; then
    echo "error: elementwise tile sweep produced no results" >&2
    return 1
  fi
  IFS=, read -r b gbps <<< "$best"
  echo "Elementwise best tile after stage 1: B=$b at ${gbps} GB/s"

  echo
  echo "Elementwise stage 2/2: optimization-hint sweep at B=$b"
  for hint in $ELEM_HINTS; do
    IFS=, read -r cta occupancy max_divisibility <<< "$hint"
    if is_default_hint "$cta" "$occupancy" "$max_divisibility"; then
      continue
    fi
    if ! hint_allowed "$cta" "$occupancy" "$max_divisibility"; then
      echo "skip elemwise B=$b hints=($cta,$occupancy,$max_divisibility): exceeds MAX_HINT=$MAX_HINT"
      continue
    fi
    run_elem_candidate "$b" "$cta" "$occupancy" "$max_divisibility"
  done
}

print_top_rows() {
  if [[ "$TARGET" == "all" || "$TARGET" == "gemm" ]]; then
    echo
    echo "GEMM tuning CSV: $GEMM_OUT"
    if [[ -s "$GEMM_OUT" ]]; then
      tmp="$(mktemp)"
      awk -F, 'NR > 1 { print $NF "\t" $0 }' "$GEMM_OUT" \
        | sort -t "$(printf '\t')" -k1,1gr \
        | cut -f2- > "$tmp"
      head -n 10 "$tmp"
      rm -f "$tmp"
    fi
  fi
  if [[ "$TARGET" == "all" || "$TARGET" == "elemwise" ]]; then
    echo
    echo "Elementwise tuning CSV: $ELEM_OUT"
    if [[ -s "$ELEM_OUT" ]]; then
      tmp="$(mktemp)"
      awk -F, 'NR > 1 { print $NF "\t" $0 }' "$ELEM_OUT" \
        | sort -t "$(printf '\t')" -k1,1gr \
        | cut -f2- > "$tmp"
      head -n 10 "$tmp"
      rm -f "$tmp"
    fi
  fi
}

echo "Rust tuning sweep: target=$TARGET sm_arch=$SM_ARCH max_hint=$MAX_HINT"
if [[ "$TARGET" == "all" || "$TARGET" == "gemm" ]]; then
  echo "GEMM variants: $GEMM_VARIANTS"
  echo "GEMM sizes: $GEMM_SIZES"
  echo "GEMM tiles: $GEMM_TILES"
  echo "GEMM tile-sweep hint triples: $GEMM_TILE_HINTS"
  echo "GEMM hint triples: $GEMM_HINTS"
  echo "GEMM hint sweep enabled: $GEMM_HINT_SWEEP"
fi
if [[ "$TARGET" == "all" || "$TARGET" == "elemwise" ]]; then
  echo "Elementwise B values: $ELEM_BS"
  echo "Elementwise hint triples: $ELEM_HINTS"
fi

build_crates
if [[ "$TARGET" == "all" || "$TARGET" == "gemm" ]]; then
  run_gemm_sweep
fi
if [[ "$TARGET" == "all" || "$TARGET" == "elemwise" ]]; then
  run_elem_sweep
fi
print_top_rows
