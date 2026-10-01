#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# scripts/smoketest.sh -- run every cuda-oxide example and report pass/fail
# per GPU-aware gating rules.
#
# Default behavior (no arguments): run every example under
# crates/rustc-codegen-cuda/examples/. Examples are categorized and a
# category-specific verdict rule is applied to the cargo output:
#
#   standard     -- execution must succeed (SUCCESS/PASS/Complete marker).
#   error        -- compilation must fail with a recognized diagnostic.
#                   Covers both intentional diagnostic fixtures and known
#                   support gaps (see STATUS.md). Signal termination is
#                   never accepted.
#   tcgen05      -- 5th-gen tensor cores; sm_100 datacenter only. On sm_100
#                   require full execution; elsewhere PTX compilation is
#                   sufficient, and when a capable ptxas is available the
#                   PTX must also assemble (the ptxas gate below).
#   wgmma        -- Hopper only (sm_90a). On Hopper require execution;
#                   elsewhere PTX compilation is sufficient (plus the
#                   ptxas gate).
#   blackwell-mma -- Blackwell-consumer-only MMA (kind::mxf8f6f4 exists on
#                   sm_120/sm_121 alone), always compiled with
#                   `--arch=sm_120a` because the generated target gating
#                   rejects every other architecture at compile time. On an
#                   sm_120/121 host require full execution; elsewhere the
#                   example verifies the generated PTX and exits 0.
#   ltoir        -- runs with `--emit-nvvm-ir --arch=<host>`; the host
#                   compute capability is detected via `nvidia-smi` so the
#                   resulting cubin actually loads. Execution must succeed
#                   — or the example may opt to `println!("skipping: ...")`
#                   with exit 0 (e.g. mathdx_ffi_test when MATHDX_ROOT is
#                   unset), in which case we require the cuda-oxide
#                   NVVM IR (`.ll`) to have been generated.
#   ltoir-modern -- like ltoir, but the example needs the modern NVVM path
#                   (small scalar externs are rejected by the legacy CUDA 12
#                   LLVM 7 dialect by design). Uses the host arch on
#                   Blackwell+ (CC >= 10.0) and requires full execution
#                   there; elsewhere compiles for the sm_100 floor and
#                   requires the NVVM IR (`.ll`) plus its `.target` sidecar.
#   auto-nvvm    -- runs without NVVM or architecture flags to check automatic
#                   libdevice and target selection. Compile-only CI supplies a
#                   target because no GPU is available.
#   iket         -- IKET-annotated kernels; the placeholder ABI needs sm_90+.
#                   On a host with CC >= 9.0, run with an explicit --arch
#                   matching the host and require the SUCCESS marker. On
#                   GPU-less or pre-9.0 hosts (and in --compile-only mode),
#                   build with the pinned sm_90 floor and require the device
#                   artifact.
#   blackwell-compile -- compile-only coverage pinned to exact sm_120a. These
#                   kernels are never launched.
#   sm100-compile -- compile-only coverage pinned to exact sm_100a. These
#                   kernels are never launched.
#   NVVM_VERIFY_EXAMPLES are compiled through the real libNVVM verifier and
#                   compiler in compile-only mode.
#
# Full-debug census:
#
#   normal examples  -> build --device-debug -> PTX/DWARF -> ptxas
#   configured cubin -> its normal cubin route -> DWARF checks
#
# It runs no kernels. Structural debug checks run; optimized shape tests do not.
#
# Categories are bash arrays at the top of this file. When adding an
# error* example, also update STATUS.md and run
# scripts/check-error-example-status.sh to verify both are in sync.
#
# See --help for runtime flags.

set -uo pipefail

# ---- Example categorization ---------------------------------------------

TCGEN05_EXAMPLES=(gemm_sol gemm_sol_final tcgen05 tcgen05_matmul)
WGMMA_EXAMPLES=(wgmma)
BLACKWELL_MMA_EXAMPLES=(mma_mxf8f6f4)
LTOIR_EXAMPLES=(addressof_sharedarray cpp_consumes_rust_device device_ffi_test legacy_atomic_fadd legacy_atomic_rmw_cas legacy_nvvm_pointer_shapes manual_launch_libdevice mathdx_ffi_test primitive_stress)
LTOIR_MODERN_EXAMPLES=(small_type_ffi_test)
AUTO_NVVM_EXAMPLES=(libdevice_math)
IKET_EXAMPLES=(iket_trace)
BLACKWELL_COMPILE_EXAMPLES=(generated_intrinsics_blackwell)
SM100_COMPILE_EXAMPLES=(redux_f32)
NVVM_VERIFY_EXAMPLES=(cp_async_small device_global enum_constant_provenance ex2_approx_f16 generated_intrinsics generated_intrinsics_blackwell generated_ldmatrix kernel_reference_validity legacy_atomic_fadd legacy_atomic_rmw_cas libdevice_math legacy_nvvm_pointer_shapes packed_atomic_add primitive_stress scoped_atomic_load_store shuffle_64 tcgen05 tuple_constant_provenance wgmma_mma_bf16)
ERROR_EXAMPLES=(error error_set_discriminant_uninhabited error_enum_bool_payload_addr error_enum_pointer_overlap error_enum_shared_pointer_layout error_heap_alloc error_host_arch_intrinsic error_host_target_feature error_kernel_shared_param error_missing_device_attr error_generated_intrinsic_abi error_generated_intrinsic_unknown_id error_generated_intrinsic_fn_pointer error_generated_intrinsic_callable)

# Per-example rustc flags policy: an example that needs a special rustc flag
# (e.g. disjoint_slice_len needs -Zinline-mir=no to keep its regression
# pattern un-inlined; reborrow needs -Zmir-opt-level=0 to keep its
# Mutability::Mut half reachable) must carry it in its own Cargo.toml via the nightly
# `profile-rustflags` feature, scoped to its own package. Never route such a
# flag through RUSTFLAGS/CARGO_ENCODED_RUSTFLAGS here: cargo keys build
# caches on rustflags, so a global flag forks a full second dependency-tree
# build (~160s on the CI runner) for that one example.

# Examples whose `main` deliberately never launches a kernel: they exist to
# prove the device code compiles, and say so in their module docs
# ("compilation and PTX generation only. Do not launch this kernel."). They
# still belong to the `standard` category because the build and the host
# binary must both succeed, but reporting a bare `PASS` would make them
# indistinguishable in the summary from an example that launched kernels and
# verified results.
NO_LAUNCH_EXAMPLES=(wgmma_mma_bf16)

# Examples whose verify-code-shape.sh asserts on `#[inline(never)]` marker
# symbols. Those markers are private, so once the middle end inlines them into
# their single caller it deletes them, and they survive only in the
# unoptimized IR. The optimized build still runs and still decides the verdict;
# these get one extra CUDA_OXIDE_NO_OPT=1 build afterwards purely to produce
# artifacts the shape check can read.
NO_OPT_SHAPE_EXAMPLES=(const_bool_dead_branch)

# This example intentionally configures a cubin artifact in Cargo metadata.
# Its supported project route necessarily includes NVVM IR + libNVVM +
# nvJitLink; the full-debug census must exercise that reality rather than
# silently rewriting the project to PTX. It is reported separately from the
# direct-LLVM census so the one accommodation cannot be mistaken for direct
# backend coverage.
FULL_DEBUG_CONFIGURED_ROUTE_EXAMPLES=(interop_cubin_identity)

classify() {
    local ex="$1" cat
    for cat in "${TCGEN05_EXAMPLES[@]}";     do [[ "$ex" == "$cat" ]] && { echo tcgen05;     return; }; done
    for cat in "${WGMMA_EXAMPLES[@]}";       do [[ "$ex" == "$cat" ]] && { echo wgmma;       return; }; done
    for cat in "${BLACKWELL_MMA_EXAMPLES[@]}"; do [[ "$ex" == "$cat" ]] && { echo blackwell-mma; return; }; done
    for cat in "${LTOIR_EXAMPLES[@]}";       do [[ "$ex" == "$cat" ]] && { echo ltoir;       return; }; done
    for cat in "${LTOIR_MODERN_EXAMPLES[@]}"; do [[ "$ex" == "$cat" ]] && { echo ltoir-modern; return; }; done
    for cat in "${AUTO_NVVM_EXAMPLES[@]}";   do [[ "$ex" == "$cat" ]] && { echo auto-nvvm;   return; }; done
    for cat in "${IKET_EXAMPLES[@]}";        do [[ "$ex" == "$cat" ]] && { echo iket;        return; }; done
    for cat in "${BLACKWELL_COMPILE_EXAMPLES[@]}"; do [[ "$ex" == "$cat" ]] && { echo blackwell-compile; return; }; done
    for cat in "${SM100_COMPILE_EXAMPLES[@]}"; do [[ "$ex" == "$cat" ]] && { echo sm100-compile; return; }; done
    for cat in "${ERROR_EXAMPLES[@]}";       do [[ "$ex" == "$cat" ]] && { echo error;       return; }; done
    echo standard
}

verify_nvvm_in_compile_only() {
    local ex="$1" candidate
    for candidate in "${NVVM_VERIFY_EXAMPLES[@]}"; do
        [[ "$ex" == "$candidate" ]] && return 0
    done
    return 1
}

example_never_launches() {
    local ex="$1" candidate
    for candidate in "${NO_LAUNCH_EXAMPLES[@]}"; do
        [[ "$ex" == "$candidate" ]] && return 0
    done
    return 1
}

full_debug_uses_configured_route() {
    local ex="$1" candidate
    for candidate in "${FULL_DEBUG_CONFIGURED_ROUTE_EXAMPLES[@]}"; do
        [[ "${ex}" == "${candidate}" ]] && return 0
    done
    return 1
}

# Return a concrete libNVVM target that satisfies both the detected/default
# target and an example's generated-intrinsic floor. Compile-only mode never
# executes the artifact, so raising an older host target here is intentional:
# this lane proves that libNVVM accepts the selected lowering.
nvvm_verify_arch() {
    local ex="$1" arch="${LTOIR_ARCH}" floor=0 number
    case "${ex}" in
        wgmma_mma_bf16)
            printf '%s\n' 'sm_90a'
            return
            ;;
        cp_async_small) floor=80 ;;
        ex2_approx_f16) floor=75 ;;
        generated_intrinsics) floor=80 ;;
        generated_ldmatrix) floor=75 ;;
        packed_atomic_add) floor=90 ;;
        shuffle_64) floor=75 ;;
    esac
    if [[ ${floor} -ne 0 && "${arch}" =~ ^sm_([0-9]+)[af]?$ ]]; then
        number=$((10#${BASH_REMATCH[1]}))
        if [[ ${number} -lt ${floor} ]]; then
            arch="sm_${floor}"
        fi
    fi
    printf '%s\n' "${arch}"
}

# ---- CLI -----------------------------------------------------------------

usage() {
    cat <<'EOF'
Usage: scripts/smoketest.sh [OPTIONS]

Run every cuda-oxide example and report PASS/FAIL per GPU-aware gating
rules. With no options, runs all examples.

OPTIONS
  -o, --only PATTERN   Run only examples whose name matches the bash regex
                       PATTERN (e.g. -o 'tcgen05|wgmma').
  -s, --skip PATTERN   Skip examples whose name matches PATTERN.
  -c, --compile-only   Compile each example without running it. Most use
                       `cargo oxide build`; designated NVVM regressions use
                       `emit-ltoir` to include real libNVVM verification.
                       Non-error categories must leave a fresh artifact;
                       every emitted .ptx must also assemble with ptxas
                       when one is available (skipped with a note when it
                       is missing or too old). Error examples must still
                       fail. Works on GPU-less CI.
      --full-debug     Compile selected examples through the normal LLVM
                       `cargo oxide build --device-debug` route (except an
                       explicitly configured cubin project, reported as a
                       separate configured-route accommodation). Validates
                       fresh artifacts, ptxas assembly, permanent
                       verify-debug-info.sh contracts, and intentional error
                       diagnostics, but skips libNVVM and optimized code-shape
                       gates because those do not accept/describe full-debug
                       output. Nothing is executed. Cannot be combined with
                       --compile-only.
  -x, --fail-fast      Stop at the first failure.
  -v, --verbose        Stream cargo output live (instead of capturing to
                       a per-example log file). Verdict is printed at the
                       end of each example.
      --keep-logs      Retain per-example logs on success as well as
                       failure. Logs for failures are always kept.
      --no-color       Disable ANSI color. Also honours the NO_COLOR env
                       var (https://no-color.org/).
  -h, --help           Show this help and exit.

POSITIONALS
  Any bare arguments are treated as additive --only patterns and joined
  with `|`. If --only is also supplied, positionals extend it (OR).

EXAMPLES
  scripts/smoketest.sh                 # run all examples
  scripts/smoketest.sh vecadd          # examples matching 'vecadd'
  scripts/smoketest.sh vecadd gemm     # matching 'vecadd' OR 'gemm'
  scripts/smoketest.sh -o '^vecadd$'   # exact-match form
  scripts/smoketest.sh -s 'wgmma|tma'  # skip wgmma and tma examples
  scripts/smoketest.sh -x -v vecadd    # stop on first fail, stream output
  scripts/smoketest.sh --compile-only  # GPU-less compile gate (used by CI)
  scripts/smoketest.sh --full-debug    # all-example full-debug compile census

Per-example logs live under .smoketest-logs/ by default. Set
SMOKETEST_LOG_DIR to override this path.
EOF
}

ONLY=""
SKIP=""
FAIL_FAST=0
VERBOSE=0
KEEP_LOGS=0
FORCE_NO_COLOR=0
COMPILE_ONLY=0
FULL_DEBUG=0
declare -a POSITIONAL=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--only)      [[ $# -lt 2 ]] && { echo "error: $1 requires a pattern" >&2; exit 2; }; ONLY="$2"; shift 2;;
        -s|--skip)      [[ $# -lt 2 ]] && { echo "error: $1 requires a pattern" >&2; exit 2; }; SKIP="$2"; shift 2;;
        -c|--compile-only) COMPILE_ONLY=1; shift;;
        --full-debug)    FULL_DEBUG=1; shift;;
        -x|--fail-fast) FAIL_FAST=1; shift;;
        -v|--verbose)   VERBOSE=1; shift;;
        --keep-logs)    KEEP_LOGS=1; shift;;
        --no-color)     FORCE_NO_COLOR=1; shift;;
        -h|--help)      usage; exit 0;;
        --)             shift; POSITIONAL+=("$@"); break;;
        -*)             echo "error: unknown option: $1" >&2; usage >&2; exit 2;;
        *)              POSITIONAL+=("$1"); shift;;
    esac
done

if [[ ${COMPILE_ONLY} -eq 1 && ${FULL_DEBUG} -eq 1 ]]; then
    echo "error: --compile-only and --full-debug are separate gates and cannot be combined" >&2
    exit 2
fi

# Bare positionals act as additive --only patterns joined with `|`.
# Combine them with any explicit --only (OR, not replace) so that
# `--only foo bar` and `-o foo bar` produce `foo|bar`.
if [[ ${#POSITIONAL[@]} -gt 0 ]]; then
    joined="$(IFS='|'; echo "${POSITIONAL[*]}")"
    if [[ -n "${ONLY}" ]]; then
        ONLY="${ONLY}|${joined}"
    else
        ONLY="${joined}"
    fi
fi

# ---- Preflight -----------------------------------------------------------

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

if [[ ! -f "${repo_root}/../Cargo.toml" ]] || [[ ! -d "crates/rustc-codegen-cuda/examples" ]]; then
    echo "error: must be run from inside the cuda-oxide repo (got ${PWD})" >&2
    exit 2
fi

if ! command -v cargo >/dev/null 2>&1; then
    echo "error: cargo not found in PATH" >&2
    exit 2
fi

if ! cargo oxide --help >/dev/null 2>&1; then
    echo "error: 'cargo oxide' subcommand missing; build it with:" >&2
    echo "         cargo build -p cargo-oxide --release" >&2
    exit 2
fi

# `CUDA_OXIDE_DEBUG` in the environment reaches every `cargo oxide` this script
# runs, so a sweep started with a full-debug value is neither a normal run nor
# the --full-debug census: mem2reg is skipped for every example, and the
# optimized code-shape gates in the normal lane then assert on output that no
# longer has the shapes they describe. #1217 measured 11 examples failing that
# way, none of them a compiler bug.
#
# --full-debug is the supported spelling of that intent -- it drives the build
# with `--device-debug`, runs the permanent verify-debug-info.sh contracts, and
# keeps the optimized shape gates out of the lane instead of failing them. So
# refuse the ambient variable rather than silently reinterpreting the sweep as
# a census the caller did not ask for: a run that quietly stopped executing
# kernels is the worse outcome of the two. Passing --full-debug as well is the
# caller asking for the mode explicitly, and stays allowed.
#
# Which values mean full debug is not decided here. `cargo oxide
# __debug-policy` reports what `DebugPolicy::parse_env_override` -- the one
# parser behind the backend and the wrapper -- makes of CUDA_OXIDE_DEBUG in
# this very environment, aliases, case and whitespace included. Any model of
# that written here would be a second implementation of the policy, and every
# shell form of it is wrong in some environment: `${v,,}` and `[[:space:]]`
# both answer to the locale, so under LC_ALL=C a plain `2` or `FULL` reads as
# an ordinary value while the compiler builds full debug; reading the parser's
# source instead trades that for a dependency on its formatting.
if ! debug_policy="$(cargo oxide __debug-policy 2>/dev/null)"; then
    echo "error: 'cargo oxide __debug-policy' failed; cannot determine device debug policy" >&2
    exit 2
fi
case "${debug_policy}" in
unset | none | line-tables | full | unrecognized) ;;
*)
    echo "error: 'cargo oxide __debug-policy' reported no usable debug policy" >&2
    echo "       (got '${debug_policy}'). This sweep asks the compiler which" >&2
    echo "       values of CUDA_OXIDE_DEBUG mean full debug rather than" >&2
    echo "       deciding for itself; rebuild cargo-oxide with:" >&2
    echo "         cargo build -p cargo-oxide --release" >&2
    exit 2
    ;;
esac

if [[ "${debug_policy}" == "full" && ${FULL_DEBUG} -eq 0 ]]; then
    echo "error: CUDA_OXIDE_DEBUG=${CUDA_OXIDE_DEBUG-} selects full device debug;" >&2
    echo "       use --full-debug instead." >&2
    echo "       The variable reaches every build this sweep runs, which skips" >&2
    echo "       dialect-mir mem2reg while the optimized code-shape gates still" >&2
    echo "       assert on optimized output (#1217). --full-debug is the mode" >&2
    echo "       that takes those gates out of the lane and checks the" >&2
    echo "       full-debug contracts instead." >&2
    exit 2
fi

# ---- Colors --------------------------------------------------------------

if [[ -t 1 ]] && [[ -z "${NO_COLOR:-}" ]] && [[ ${FORCE_NO_COLOR} -eq 0 ]]; then
    C_PASS=$'\e[32m'; C_FAIL=$'\e[31m'; C_SKIP=$'\e[33m'
    C_DIM=$'\e[2m'; C_BOLD=$'\e[1m'; C_RESET=$'\e[0m'
else
    C_PASS=""; C_FAIL=""; C_SKIP=""; C_DIM=""; C_BOLD=""; C_RESET=""
fi

# ---- Banner --------------------------------------------------------------

git_head="$(git rev-parse --short HEAD 2>/dev/null || echo '?')"
git_branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
# nvidia-smi can be present yet broken (driver mismatch, sandboxes,
# containers) and it prints its failure text to STDOUT, so trust it only
# when it exits 0 AND the compute capability parses. One probe feeds both
# the banner and the LTOIR arch so they can never disagree.
host_cc=""
if gpu_query="$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null)"; then
    gpu_info="$(head -1 <<<"${gpu_query}")"
    host_cc="$(awk -F', *' '{print $2}' <<<"${gpu_info}" | tr -d '[:space:]')"
else
    gpu_info='no GPU detected'
fi

# Detect host compute capability for the ltoir category: the cubin produced
# by the LTOIR linker must match the GPU's arch or it refuses to load.
# nvidia-smi reports something like "12.0" → sm_120, "10.0" → sm_100.
if [[ "${host_cc}" =~ ^[0-9]+\.[0-9]+$ ]]; then
    # Strip the dot: 12.0 -> 120
    LTOIR_ARCH="sm_${host_cc//./}"
else
    # No working GPU detected. ltoir examples will likely fail to execute,
    # but pick a sensible floor so the cuda-oxide side still compiles.
    LTOIR_ARCH="sm_90"
fi

# ltoir-modern examples need the modern NVVM path (sm_100+). On a Blackwell+
# host (CC >= 10.0) target the host arch so the cubin loads and the example
# executes; elsewhere compile for the sm_100 floor and expect the verdict to
# fall back to compile-only semantics.
LTOIR_MODERN_EXEC=0
if [[ "${host_cc}" =~ ^([0-9]+)\.[0-9]+$ ]] && [[ $((10#${BASH_REMATCH[1]})) -ge 10 ]]; then
    LTOIR_MODERN_ARCH="${LTOIR_ARCH}"
    LTOIR_MODERN_EXEC=1
else
    LTOIR_MODERN_ARCH="sm_100"
fi

# iket examples need the sm_90+ placeholder ABI. On a CC >= 9.0 host, target
# the host arch explicitly and require full execution; on GPU-less or pre-9.0
# hosts, compile for the pinned sm_90 floor and require only the artifact.
IKET_EXEC=0
IKET_ARCH="sm_90"
if [[ "${host_cc}" =~ ^([0-9]+)\.[0-9]+$ ]] && [[ $((10#${BASH_REMATCH[1]})) -ge 9 ]]; then
    IKET_ARCH="${LTOIR_ARCH}"
    IKET_EXEC=1
fi

# ---- ptxas gate ------------------------------------------------------------
# Compile-only verdicts accepted "a .ptx exists". That bar misses PTX that
# llc emits but ptxas rejects: e.g. `.global` initializers referencing
# `.shared` symbols, which fail driver JIT with CUDA_ERROR_INVALID_PTX only
# on real hardware. When a ptxas is available, actually assemble the PTX for
# the arch recorded in its `.target` line. In --compile-only mode this
# applies to EVERY example that emitted a .ptx (nothing executes there, so
# assembly is the strongest available check); in normal mode only the
# GPU-gated categories (tcgen05, wgmma, blackwell-mma) use it, since real
# execution supersedes assembly for everything else. Resolution order
# matches how the examples themselves shell out to ptxas: explicit override,
# PATH, then the CUDA toolkit install locations.
PTXAS_BIN=""
for ptxas_candidate in "${CUDA_OXIDE_PTXAS:-}" \
                       "$(command -v ptxas 2>/dev/null)" \
                       "${CUDA_HOME:+${CUDA_HOME}/bin/ptxas}" \
                       /usr/local/cuda/bin/ptxas \
                       /usr/local/cuda-*/bin/ptxas; do
    if [[ -n "${ptxas_candidate}" && -x "${ptxas_candidate}" ]]; then
        PTXAS_BIN="${ptxas_candidate}"
        break
    fi
done

# Assemble ${1} (a .ptx file) with the resolved ptxas for the arch its
# `.target` line records. Sets PTXAS_NOTE for the verdict text. Returns 1
# only for a real assembly failure; a missing/too-old ptxas or an arch it
# does not know is a documented skip, not a failure.
PTXAS_NOTE=""
ptxas_verify() {
    local ptx="$1" arch stderr_out
    if [[ -z "${PTXAS_BIN}" ]]; then
        PTXAS_NOTE="ptxas gate skipped: no ptxas found"
        return 0
    fi
    if [[ ! -s "${ptx}" ]]; then
        PTXAS_NOTE="ptxas gate skipped: no PTX at ${ptx}"
        return 0
    fi
    arch="$(sed -nE 's/^\.target[[:space:]]+(sm_[0-9]+[af]?).*/\1/p' "${ptx}" | head -1)"
    if [[ -z "${arch}" ]]; then
        PTXAS_NOTE="ptxas gate skipped: no .target line in ${ptx}"
        return 0
    fi
    if stderr_out="$("${PTXAS_BIN}" -arch="${arch}" "${ptx}" -o /dev/null 2>&1)"; then
        PTXAS_NOTE="ptxas ${arch} ok"
        return 0
    fi
    # An arch this ptxas predates (or a PTX ISA newer than it parses) is a
    # tooling gap, not a compiler bug: note the skip and keep the verdict.
    if grep -qE "is not defined for option 'gpu-name'|Unsupported \.version" <<<"${stderr_out}"; then
        PTXAS_NOTE="ptxas gate skipped: ${PTXAS_BIN} is too old for ${arch}"
        return 0
    fi
    PTXAS_NOTE="ptxas -arch=${arch} rejected the PTX: $(head -2 <<<"${stderr_out}" | tr '\n' ' ')"
    return 1
}

printf "%scuda-oxide smoketest%s @ %s%s%s (%s)\n" "${C_BOLD}" "${C_RESET}" "${C_BOLD}" "${git_head}" "${C_RESET}" "${git_branch}"
printf "GPU: %s\n" "${gpu_info}"
printf "LTOIR arch: %s (modern: %s)\n" "${LTOIR_ARCH}" "${LTOIR_MODERN_ARCH}"
if [[ ${COMPILE_ONLY} -eq 1 ]]; then
    printf "Mode: compile-only (device artifacts only; nothing is executed)\n"
elif [[ ${FULL_DEBUG} -eq 1 ]]; then
    printf "Mode: full-debug compile census (direct LLVM plus explicit configured routes; nothing is executed)\n"
fi
if [[ -n "${ONLY}" ]]; then printf "Filter --only: %s\n" "${ONLY}"; fi
if [[ -n "${SKIP}" ]]; then printf "Filter --skip: %s\n" "${SKIP}"; fi
echo ""

# ---- Example selection ---------------------------------------------------

# Read loop rather than `mapfile`: that builtin arrived in bash 4, and macOS
# still ships bash 3.2 as /bin/bash, where this line ended the run outright.
ALL_EXAMPLES=()
while IFS= read -r example_name; do
    ALL_EXAMPLES+=("${example_name}")
done < <(
    cd crates/rustc-codegen-cuda/examples
    for manifest in */Cargo.toml; do
        [[ -e "${manifest}" ]] || continue
        echo "${manifest%/Cargo.toml}"
    done | sort
)

# An example dir without a top-level Cargo.toml would be skipped by the
# glob above and silently shrink coverage (e.g. a restructure that nests
# the manifest one level down). Fail loudly instead.
for dir in crates/rustc-codegen-cuda/examples/*/; do
    if [[ ! -f "${dir}Cargo.toml" ]]; then
        echo "error: ${dir} has no top-level Cargo.toml; every directory under" >&2
        echo "       crates/rustc-codegen-cuda/examples/ must be an example crate" >&2
        exit 2
    fi
done

selected=()
for ex in "${ALL_EXAMPLES[@]}"; do
    if [[ -n "${ONLY}" ]] && ! [[ "${ex}" =~ ${ONLY} ]]; then continue; fi
    if [[ -n "${SKIP}" ]] &&   [[ "${ex}" =~ ${SKIP} ]]; then continue; fi
    selected+=("${ex}")
done

total=${#selected[@]}
if [[ ${total} -eq 0 ]]; then
    echo "error: no examples matched the given filters" >&2
    exit 1
fi

# ---- Verdict functions ---------------------------------------------------
#
# Each verdict_* function consumes a log file + exit code, prints the
# classification string to stdout, and returns 0 (pass) or 1 (fail).
# They never run cargo themselves; that is the caller's job.

verdict_standard() {
    local log="$1" ec="$2" ex="${3:-}"
    if [[ ${ec} -gt 128 ]]; then echo "FAIL (crashed, signal $((ec - 128)))"; return 1; fi
    if [[ ${ec} -ne 0 ]]; then   echo "FAIL (exit=${ec})";                    return 1; fi
    if grep_failure_markers "${log}"; then
        echo "FAIL (failure marker in output)"
        return 1
    fi
    # `skipping:` is an explicit, graceful opt-out (e.g. cluster on
    # pre-Hopper, mathdx_ffi_test with no MathDx SDK). Accept it as PASS so
    # standard-category examples can gate themselves on hardware/SDK presence
    # without having to fake a success marker.
    #
    # This has to recognise the declaration in every spelling the examples
    # use, because the success-marker check below matches `SUCCESS|PASS|
    # Complete` anywhere in the log and skip messages routinely contain those
    # words. Missing a skip here therefore does not merely lose the
    # "(skipped)" annotation: it promotes the example to a full execution
    # PASS, indistinguishable from one that launched kernels and checked
    # results. That is how `Skipping: ... -- PASS (skipped)` used to report a
    # clean PASS, and `generated_ldmatrix` still prints the
    # `PASS (skipped): ...` form below sm_75.
    if grep -qiE '^[[:space:]]*(skipping:|pass \(skipped\))' "${log}"; then
        echo "PASS (skipped)"
        return 0
    fi
    if grep -qE 'SUCCESS|PASS|Complete' "${log}"; then
        if example_never_launches "${ex}"; then
            echo "PASS (compiled, no launch)"
        else
            echo "PASS"
        fi
        return 0
    fi
    echo "FAIL (no success marker)"
    return 1
}

# Returns 0 iff the log contains any of our known failure signals. Designed
# to be aggressive about false negatives — we'd rather flag a borderline
# example for human review than miss a regression. Each pattern is anchored
# enough not to match incidental prose ("This will fail if...", or the
# success line "(no assertion failed)" some examples print).
grep_failure_markers() {
    local log="$1"
    grep -qE \
        -e '(^|[[:space:]])FAIL(ED)?($|[[:space:]:!.])' \
        -e '(^|[[:space:]])✗($|[[:space:]])' \
        -e 'panicked at |thread .* panicked' \
        -e 'assertion failed:|assertion `[^`]*` failed' \
        -e 'illegal memory access|invalid argument|misaligned address' \
        -e 'CUDA(_ERROR)?[ _][A-Z_]*(FAIL|ERROR)' \
        -e 'fatal( error)?:' \
        "${log}"
}

verdict_error() {
    local log="$1" ec="$2" ex="$3"
    if [[ ${ec} -gt 128 ]]; then echo "FAIL (crashed, signal $((ec - 128)))"; return 1; fi
    # Every error fixture promises a rejected build, not merely a familiar
    # diagnostic. Check status before any fixture-specific marker can return
    # success: a downgraded warning with exit 0 is a regression.
    if [[ ${ec} -eq 0 ]]; then
        echo "FAIL (compilation succeeded, expected failure)"
        return 1
    fi

    # The generated-intrinsic fixtures protect fail-closed compiler contracts,
    # so merely observing an unrelated compile error is not enough.
    case "${ex}" in
        error_enum_bool_payload_addr)
            if ! grep -Fq 'canonical storage type' "${log}" \
                || ! grep -Fq 'a borrow that escapes into a call keeps no such rewrite and is refused here' "${log}"; then
                echo "FAIL (missing canonical-storage payload-address diagnostic)"
                return 1
            fi
            ;;
        error_enum_pointer_overlap)
            if ! grep -Fq 'overlapping pointer and non-identical storage' "${log}"; then
                echo "FAIL (missing overlapping enum pointer-provenance diagnostic)"
                return 1
            fi
            ;;
        error_enum_shared_pointer_layout)
            if ! grep -Fq 'arrays containing shared-memory pointers are not supported' "${log}"; then
                echo "FAIL (missing shared-pointer array layout diagnostic)"
                return 1
            fi
            ;;
        error_kernel_shared_param)
            if ! grep -Fq 'is a pointer into shared memory' "${log}"; then
                echo "FAIL (missing shared-memory kernel-parameter diagnostic)"
                return 1
            fi
            ;;
        error_generated_intrinsic_abi)
            if ! grep -Fq 'cuda-intrinsics ABI mismatch' "${log}" \
                || ! grep -Fq '__cuda_oxide_intrinsic_abi_v2::i0001' "${log}"; then
                echo "FAIL (missing intrinsic ABI-v2 diagnostic)"
                return 1
            fi
            ;;
        error_generated_intrinsic_unknown_id)
            if ! grep -Fq 'cuda-intrinsics ABI mismatch' "${log}" \
                || ! grep -Fq '__cuda_oxide_intrinsic_abi_v1::i9999' "${log}"; then
                echo "FAIL (missing unknown intrinsic-ID diagnostic)"
                return 1
            fi
            ;;
        error_generated_intrinsic_fn_pointer)
            if ! grep -Fq 'must be called directly and cannot be converted to a function pointer' "${log}"; then
                echo "FAIL (missing function-pointer intrinsic diagnostic)"
                return 1
            fi
            ;;
        error_generated_intrinsic_callable)
            if ! grep -Fq 'generated CUDA intrinsics require direct call-site lowering' "${log}"; then
                echo "FAIL (missing direct-call-only intrinsic diagnostic)"
                return 1
            fi
            ;;
        # These three pin an exact diagnostic in their own source or README
        # ("Expected: the build FAILS with this exact diagnostic (pinned)"),
        # so an unrelated compile error must not satisfy them either.
        error_heap_alloc)
            if ! grep -Fq 'heap allocation is not supported in kernels' "${log}"; then
                echo "FAIL (missing heap-allocation diagnostic)"
                return 1
            fi
            ;;
        error_host_target_feature)
            if ! grep -Fq 'requires host CPU target features' "${log}"; then
                echo "FAIL (missing host-CPU target-feature diagnostic)"
                return 1
            fi
            ;;
        error_host_arch_intrinsic)
            # x86_64-only fixture: `_rdtsc` is the common intrinsic with no
            # `#[target_feature]`. On other hosts it refuses to build through
            # its own compile_error!, which is the accepted outcome there.
            if [[ "$(uname -m)" == "x86_64" ]]; then
                if ! grep -Fq 'intrinsic for the host CPU' "${log}"; then
                    echo "FAIL (missing host-CPU intrinsic diagnostic)"
                    return 1
                fi
            elif ! grep -Fq 'x86_64-only fixture' "${log}"; then
                echo "FAIL (missing x86_64-only compile_error marker)"
                return 1
            fi
            ;;
        error_missing_device_attr)
            if ! grep -Fq 'only works inside `#[kernel]` / `#[device]`' "${log}"; then
                echo "FAIL (missing host-only-stub diagnostic)"
                return 1
            fi
            ;;
        error_set_discriminant_uninhabited)
            if ! grep -Fq 'cannot select uninhabited variant' "${log}"; then
                echo "FAIL (missing uninhabited-variant diagnostic)"
                return 1
            fi
            ;;
        # `error` stays on the generic check below: unlike the fixtures above it
        # pins no single message, carrying two kernels to show that a valid one
        # still compiles while `core::fmt` machinery is refused.
    esac

    if grep -qE 'Device codegen failed|Translation failed|Compilation error|Unsupported construct' "${log}"; then
        echo "PASS (expected compile failure)"
        return 0
    fi
    # Non-zero exit + a real rustc/cargo error line is still a legit
    # "example refused to compile" signal. Crucially, we require the error
    # line: just exit=42 with no diagnostic is NOT accepted (unlike the old
    # CLAUDE.md blob).
    if [[ ${ec} -ne 0 ]] && grep -qE '^error(\[|:)|error: could not compile|error aborting due to' "${log}"; then
        echo "PASS (expected compile failure, exit=${ec})"
        return 0
    fi
    echo "FAIL (exit=${ec} but no compile-error marker)"
    return 1
}

verdict_tcgen05() {
    local ex="$1" log="$2" ec="$3"
    if [[ ${ec} -gt 128 ]]; then echo "FAIL (crashed, signal $((ec - 128)))"; return 1; fi
    if grep -qE 'WARNING: tcgen05 requires|Skipping GPU test: requires sm_100|Skipping benchmark: requires sm_100|tcgen05 \(5th gen tensor cores\) requires sm_100|PTX was generated successfully' "${log}"; then
        if grep -qE 'PTX written|PTX Verification|PTX file generated' "${log}"; then
            if ! ptxas_verify "crates/rustc-codegen-cuda/examples/${ex}/${ex//-/_}.ptx"; then
                echo "FAIL (tcgen05, ${PTXAS_NOTE})"
                return 1
            fi
            echo "PASS (tcgen05, PTX compiled; ${PTXAS_NOTE})"
            return 0
        fi
        echo "FAIL (tcgen05, PTX not generated)"
        return 1
    fi
    if [[ ${ec} -ne 0 ]]; then echo "FAIL (tcgen05, exit=${ec})"; return 1; fi
    if grep_failure_markers "${log}"; then
        echo "FAIL (tcgen05, failure marker in output)"
        return 1
    fi
    if grep -qE 'SUCCESS|PASS|Complete' "${log}"; then echo "PASS (tcgen05, executed)"; return 0; fi
    echo "FAIL (tcgen05, no success marker)"
    return 1
}

verdict_wgmma() {
    local ex="$1" log="$2" ec="$3"
    if [[ ${ec} -gt 128 ]]; then echo "FAIL (crashed, signal $((ec - 128)))"; return 1; fi
    if grep -qE 'WARNING: WGMMA requires|WGMMA is Hopper-only|PTX load failed \(expected on non-Hopper\)|PTX module loaded' "${log}"; then
        if grep -qE 'PTX written|PTX Verification|PTX file generated|inspect generated PTX|\.ptx' "${log}"; then
            if ! ptxas_verify "crates/rustc-codegen-cuda/examples/${ex}/${ex//-/_}.ptx"; then
                echo "FAIL (wgmma, ${PTXAS_NOTE})"
                return 1
            fi
            echo "PASS (wgmma, PTX compiled; ${PTXAS_NOTE})"
            return 0
        fi
        echo "FAIL (wgmma, PTX not generated)"
        return 1
    fi
    if [[ ${ec} -ne 0 ]]; then echo "FAIL (wgmma, exit=${ec})"; return 1; fi
    if grep_failure_markers "${log}"; then
        echo "FAIL (wgmma, failure marker in output)"
        return 1
    fi
    if grep -qE 'SUCCESS|PASS|Complete' "${log}"; then echo "PASS (wgmma, executed)"; return 0; fi
    echo "FAIL (wgmma, no success marker)"
    return 1
}

verdict_blackwell_mma() {
    local ex="$1" log="$2" ec="$3"
    if [[ ${ec} -gt 128 ]]; then echo "FAIL (crashed, signal $((ec - 128)))"; return 1; fi
    # Non-sm_120/121 host: the example declares the skip and must still
    # prove the sm_120a PTX was generated with the block-scaled instruction.
    if grep -qE 'mxf8f6f4 block-scale MMA requires sm_120' "${log}"; then
        if [[ ${ec} -eq 0 ]] && grep -qE 'PTX was generated successfully' "${log}"; then
            if ! ptxas_verify "crates/rustc-codegen-cuda/examples/${ex}/${ex//-/_}.ptx"; then
                echo "FAIL (blackwell-mma, ${PTXAS_NOTE})"
                return 1
            fi
            echo "PASS (blackwell-mma, PTX compiled; ${PTXAS_NOTE})"
            return 0
        fi
        echo "FAIL (blackwell-mma, PTX not generated)"
        return 1
    fi
    if [[ ${ec} -ne 0 ]]; then echo "FAIL (blackwell-mma, exit=${ec})"; return 1; fi
    if grep_failure_markers "${log}"; then
        echo "FAIL (blackwell-mma, failure marker in output)"
        return 1
    fi
    if grep -qE 'SUCCESS|PASS|Complete' "${log}"; then echo "PASS (blackwell-mma, executed)"; return 0; fi
    echo "FAIL (blackwell-mma, no success marker)"
    return 1
}

verdict_ltoir() {
    local ex="$1" log="$2" ec="$3"
    # Hyphens in example names become underscores in the crate-named
    # artifact (see verdict_compile).
    local ll_file="crates/rustc-codegen-cuda/examples/${ex}/${ex//-/_}.ll"
    if [[ ${ec} -gt 128 ]]; then echo "FAIL (crashed, signal $((ec - 128)))"; return 1; fi
    # `skipping:` marker -- the example opted out (e.g. mathdx_ffi_test with
    # MATHDX_ROOT unset). Accept as pass so long as the cuda-oxide side
    # still produced NVVM IR.
    if grep -qE '^[[:space:]]*skipping:' "${log}"; then
        if [[ -f "${ll_file}" ]]; then
            echo "PASS (LTOIR, skipped: NVVM IR generated)"
            return 0
        fi
        echo "FAIL (LTOIR, skipped but no NVVM IR)"
        return 1
    fi
    if [[ ${ec} -ne 0 ]]; then   echo "FAIL (LTOIR, exit=${ec})";             return 1; fi
    if grep_failure_markers "${log}"; then
        echo "FAIL (LTOIR, failure marker in output)"
        return 1
    fi
    if grep -qE 'SUCCESS|PASS|Complete|NVVM IR is ready' "${log}"; then
        echo "PASS (LTOIR)"
        return 0
    fi
    echo "FAIL (LTOIR, no success marker)"
    return 1
}

verdict_ltoir_modern() {
    local ex="$1" log="$2" ec="$3"
    local ex_dir="crates/rustc-codegen-cuda/examples/${ex}"
    local artifact="${ex//-/_}"
    if [[ ${ec} -gt 128 ]]; then echo "FAIL (crashed, signal $((ec - 128)))"; return 1; fi
    if [[ ${LTOIR_MODERN_EXEC} -eq 1 ]]; then
        # Blackwell+ host: the cubin targets the host arch, so the example
        # must execute end to end like any other ltoir example.
        if [[ ${ec} -ne 0 ]]; then echo "FAIL (LTOIR modern, exit=${ec})"; return 1; fi
        if grep_failure_markers "${log}"; then
            echo "FAIL (LTOIR modern, failure marker in output)"
            return 1
        fi
        if grep -qE 'SUCCESS|PASS|Complete' "${log}"; then
            echo "PASS (LTOIR modern, executed)"
            return 0
        fi
        echo "FAIL (LTOIR modern, no success marker)"
        return 1
    fi
    # Pre-Blackwell or GPU-less host: the sm_100-floor cubin cannot load, so
    # require the compile half only -- fresh NVVM IR plus its .target sidecar
    # (mirrors verdict_compile's NVVM-IR artifact rule).
    if [[ -s "${ex_dir}/${artifact}.ll" && -s "${ex_dir}/${artifact}.target" ]]; then
        echo "PASS (LTOIR modern, NVVM IR compiled for ${LTOIR_MODERN_ARCH})"
        return 0
    fi
    echo "FAIL (LTOIR modern, no NVVM IR for the ${LTOIR_MODERN_ARCH} floor)"
    return 1
}

verdict_iket() {
    local ex="$1" log="$2" ec="$3"
    local ex_dir="crates/rustc-codegen-cuda/examples/${ex}"
    local artifact="${ex//-/_}"
    if [[ ${ec} -gt 128 ]]; then echo "FAIL (crashed, signal $((ec - 128)))"; return 1; fi
    if [[ ${IKET_EXEC} -eq 1 ]]; then
        # CC >= 9.0 host: the kernel targeted the host arch and must execute.
        if [[ ${ec} -ne 0 ]]; then echo "FAIL (iket, exit=${ec})"; return 1; fi
        if grep_failure_markers "${log}"; then
            echo "FAIL (iket, failure marker in output)"
            return 1
        fi
        if grep -qE 'SUCCESS|PASS|Complete' "${log}"; then
            echo "PASS (iket, executed on ${IKET_ARCH})"
            return 0
        fi
        echo "FAIL (iket, no success marker)"
        return 1
    fi
    # GPU-less or pre-9.0 host: run_cargo built for the sm_90 floor instead;
    # the bar is a clean build plus a fresh device artifact.
    if [[ ${ec} -ne 0 ]]; then echo "FAIL (iket, exit=${ec})"; return 1; fi
    if [[ -s "${ex_dir}/${artifact}.ptx" || -s "${ex_dir}/${artifact}.ll" ]]; then
        echo "PASS (iket, compiled for ${IKET_ARCH})"
        return 0
    fi
    echo "FAIL (iket, no device artifact for the ${IKET_ARCH} floor)"
    return 1
}

# Artifact verdict, used for every non-error category in --compile-only and
# --full-debug modes. Two requirements:
#   1. `cargo oxide build` exited 0. Device codegen failures are rustc
#      fatals (see rustc-codegen-cuda/src/lib.rs join on device results),
#      so a broken device pipeline cannot exit 0.
#   2. A fresh device artifact exists: {ex}.ptx, or {ex}.ll plus a concrete
#      {ex}.target for the NVVM-IR path. cargo-oxide deletes stale ones before
#      building (clean_generated_files), so presence proves this build emitted
#      them. This catches collector regressions where the build "succeeds"
#      because no #[kernel] was found and device codegen never ran, and prevents
#      target-specific NVVM IR from being published without its source target.
# Interop examples write PTX into their configured ptx_dir instead, and
# cargo-oxide itself verifies that file exists (exits non-zero if not),
# so the exit code alone is trusted for them.
verdict_compile() {
    local ex="$1" log="$2" ec="$3"
    local ex_dir="crates/rustc-codegen-cuda/examples/${ex}"
    # Artifacts are named after the crate, and cargo normalizes hyphens
    # to underscores (rustlantis-smoke emits rustlantis_smoke.ptx). This
    # assumes dir name == package name, which holds for every example and
    # is equally assumed by cargo-oxide's clean_generated_files; a renamed
    # package fails here loudly rather than passing on a stale artifact.
    local artifact="${ex//-/_}"
    if [[ ${ec} -gt 128 ]]; then echo "FAIL (crashed, signal $((ec - 128)))"; return 1; fi
    if [[ ${ec} -ne 0 ]]; then   echo "FAIL (exit=${ec})";                    return 1; fi
    if [[ ${FULL_DEBUG} -eq 1 ]]; then
        case "${ex}" in
            device_ffi_test|mathdx_ffi_test|small_type_ffi_test)
                # These fixtures deliberately leave CUDA device functions
                # undefined: their normal route links the Rust artifact with
                # external LTOIR later.  An executable ptxas link must fail,
                # so validate a relocatable object and its exact undefined
                # interface instead.
                full_debug_relocatable_ffi_verdict \
                    "${ex}" "${ex_dir}/${artifact}.ptx"
                return
                ;;
            cutile_inter_kernel)
                full_debug_ptx_verdict \
                    "${ex_dir}/simt/cutile_inter_kernel_simt.ptx" \
                    "nested interop PTX"
                return
                ;;
            interop_cubin_identity)
                full_debug_cubin_verdict "${ex_dir}/device/scale_offset_device"
                return
                ;;
            *)
                # A direct full-debug build must finish as debug PTX. Do not
                # accept the generic .ll/.target fallback used by compile-only:
                # that would hide a route change to NVVM IR or dropped debug
                # propagation across almost the entire census.
                full_debug_ptx_verdict "${ex_dir}/${artifact}.ptx" "direct LLVM PTX"
                return
                ;;
        esac
    fi
    if [[ ${FULL_DEBUG} -eq 0 ]] && verify_nvvm_in_compile_only "${ex}"; then
        if [[ -s "${ex_dir}/${artifact}.ll" && -s "${ex_dir}/${artifact}.ltoir" ]]; then
            # An example may also have produced a .ptx via its direct-LLVM
            # invocation (e.g. tcgen05's dual-route branch); that artifact
            # must assemble like any other compile-only PTX.
            if [[ -s "${ex_dir}/${artifact}.ptx" ]] \
                && ! ptxas_verify "${ex_dir}/${artifact}.ptx"; then
                echo "FAIL (${PTXAS_NOTE})"
                return 1
            fi
            echo "PASS (verified and compiled by libNVVM)"
            return 0
        fi
        echo "FAIL (libNVVM compile produced no fresh LTOIR)"
        return 1
    fi
    if [[ -s "${ex_dir}/${artifact}.ptx" ]]; then
        # Nothing executes in compile-only mode, so every emitted PTX must
        # at least assemble: driver-JIT-only errors (e.g. .shared symbols
        # in .global initializers) otherwise pass CI silently.
        if ! ptxas_verify "${ex_dir}/${artifact}.ptx"; then
            echo "FAIL (${PTXAS_NOTE})"
            return 1
        fi
        echo "PASS (compiled; ${PTXAS_NOTE})"
        return 0
    fi
    if [[ -s "${ex_dir}/${artifact}.ll" ]]; then
        local target_file="${ex_dir}/${artifact}.target"
        local target
        target="$(sed -n '1p' "${target_file}" 2>/dev/null)"
        if [[ "${target}" =~ ^sm_[0-9]+[af]?$ ]]; then
            if grep -qx 'compile-options=v1' "${target_file}" && \
               [[ ! -s "${ex_dir}/${artifact}.options" ]]; then
                echo "FAIL (versioned NVVM IR target is missing its .options sidecar)"
                return 1
            fi
            echo "PASS (compiled NVVM IR for ${target})"
            return 0
        fi
        echo "FAIL (NVVM IR emitted without a concrete .target sidecar)"
        return 1
    fi
    # Match only the real interop config shapes ([[package.metadata.
    # cuda-oxide.device-crates]] tables or a device-crates = [...] key),
    # not the substring anywhere: a stray comment must not silently
    # exempt an example from the artifact check.
    if grep -qE '^[[:space:]]*(\[\[package\.metadata\.cuda-oxide\.device-crates\]\]|device-crates[[:space:]]*=)' \
        "${ex_dir}/Cargo.toml" 2>/dev/null; then
        echo "PASS (compiled, interop)"
        return 0
    fi
    echo "FAIL (built, but no device artifact emitted)"
    return 1
}

# Assert the artifact is full-debug PTX before asking ptxas to consume it.
# Merely finding a fresh .ptx file is insufficient: a lost --device-debug
# request still emits valid optimized PTX and would make the census useless.
full_debug_ptx_verdict() {
    local ptx="$1" route="$2"
    if [[ ! -s "${ptx}" ]]; then
        echo "FAIL (${route} missing at ${ptx})"
        return 1
    fi
    if ! grep -Eq '^\.target[[:space:]]+.*,[[:space:]]*debug([[:space:]]*,.*)?$' "${ptx}"; then
        echo "FAIL (${route} is not marked debug in its .target directive)"
        return 1
    fi
    if ! grep -Eq '^[[:space:]]*\.section[[:space:]]+\.debug_info([[:space:]]|$)' "${ptx}"; then
        echo "FAIL (${route} has no .debug_info section)"
        return 1
    fi
    if [[ -z "${PTXAS_BIN}" ]]; then
        echo "FAIL (${route} was not assembled: no ptxas found)"
        return 1
    fi
    if ! ptxas_verify "${ptx}"; then
        echo "FAIL (${route}; ${PTXAS_NOTE})"
        return 1
    fi
    if [[ "${PTXAS_NOTE}" == "ptxas gate skipped:"* ]]; then
        echo "FAIL (${route} was not assembled; ${PTXAS_NOTE})"
        return 1
    fi
    echo "PASS (full-debug ${route}; ${PTXAS_NOTE})"
}

# These direct-LLVM FFI examples call functions supplied by external CUDA
# LTOIR modules. `ptxas foo.ptx` tries to link an executable image and must
# reject those undefined functions; `ptxas -c` is the correct consumability
# check. Require a nonempty relocatable ELF, retained DWARF, and the fixture's
# exact undefined interface so a new accidental reference cannot pass.
full_debug_relocatable_ffi_verdict() {
    local ex="$1" ptx="$2"
    local arch object elf_reader sections dwarf actual_undefined expected_undefined stderr_out

    case "${ex}" in
        device_ffi_test)
            expected_undefined=$'char_store\nchar_to_upper\ncub_warp_reduce_sum_f32\ndot_product\nfast_rsqrt\nmagnitude_squared\nsimple_add\nsmem_get_base_addr\nsmem_read_aligned_128\nsmem_write_aligned_128\nvprintf\nwarp_ballot\nwarp_reduce_sum'
            ;;
        mathdx_ffi_test)
            expected_undefined=$'cublasdx_gemm_32x32x32_block_dim_x\ncublasdx_gemm_32x32x32_block_dim_y\ncublasdx_gemm_32x32x32_block_dim_z\ncublasdx_gemm_32x32x32_f32\ncublasdx_gemm_32x32x32_f32_alphabeta\ncublasdx_gemm_32x32x32_smem_size\ncublasdx_gemm_32x32x32_smem_size_ab\ncufftdx_fft_16_c2c_f32_forward\ncufftdx_fft_16_c2c_f32_inverse\ncufftdx_fft_16_elements_per_thread\ncufftdx_fft_16_storage_size\ncufftdx_fft_32_storage_size\ncufftdx_fft_8_c2c_f32_forward\ncufftdx_fft_8_c2c_f32_inverse\ncufftdx_fft_8_elements_per_thread\ncufftdx_fft_8_storage_size\ndebug_extern_double_array'
            ;;
        small_type_ffi_test)
            expected_undefined=$'small_add_u16\nsmall_add_u8\nsmall_half_add\nsmall_not_bool\nsmall_scale_i16\nsmall_scale_i8\nsmall_widen_i8\nsmall_widen_u16'
            ;;
        *)
            echo "FAIL (unrecognized relocatable FFI fixture ${ex})"
            return 1
            ;;
    esac

    if [[ ! -s "${ptx}" ]]; then
        echo "FAIL (full-debug FFI PTX missing at ${ptx})"
        return 1
    fi
    if ! grep -Eq '^\.target[[:space:]]+.*,[[:space:]]*debug([[:space:]]*,.*)?$' "${ptx}" \
        || ! grep -Eq '^[[:space:]]*\.section[[:space:]]+\.debug_info([[:space:]]|$)' "${ptx}"; then
        echo "FAIL (full-debug FFI PTX is missing its debug target/metadata)"
        return 1
    fi
    if [[ -z "${PTXAS_BIN}" ]]; then
        echo "FAIL (full-debug FFI PTX was not assembled: no ptxas found)"
        return 1
    fi
    arch="$(sed -nE 's/^\.target[[:space:]]+(sm_[0-9]+[af]?).*/\1/p' "${ptx}" | head -1)"
    if [[ -z "${arch}" ]]; then
        echo "FAIL (full-debug FFI PTX has no concrete .target architecture)"
        return 1
    fi
    object="$(mktemp "${TMPDIR:-/tmp}/cuda-oxide-full-debug-${ex}.XXXXXX.o")" || {
        echo "FAIL (could not allocate a temporary FFI object)"
        return 1
    }
    if ! stderr_out="$("${PTXAS_BIN}" -arch="${arch}" -c "${ptx}" -o "${object}" 2>&1)"; then
        rm -f "${object}"
        echo "FAIL (ptxas -c -arch=${arch} rejected full-debug FFI PTX: $(head -2 <<<"${stderr_out}" | tr '\n' ' '))"
        return 1
    fi
    if [[ ! -s "${object}" ]]; then
        rm -f "${object}"
        echo "FAIL (ptxas -c emitted an empty full-debug FFI object)"
        return 1
    fi

    elf_reader="$(command -v readelf || command -v llvm-readelf || true)"
    if [[ -z "${elf_reader}" ]]; then
        rm -f "${object}"
        echo "FAIL (no readelf/llvm-readelf available to inspect full-debug FFI object)"
        return 1
    fi
    sections="$("${elf_reader}" -W -S "${object}" 2>/dev/null)" || {
        rm -f "${object}"
        echo "FAIL (could not read full-debug FFI object sections)"
        return 1
    }
    if ! grep -q '[.]debug_info' <<<"${sections}" \
        || ! grep -q '[.]debug_line' <<<"${sections}"; then
        rm -f "${object}"
        echo "FAIL (relocatable full-debug FFI object is missing .debug_info/.debug_line)"
        return 1
    fi
    dwarf="$("${elf_reader}" --debug-dump=info "${object}" 2>/dev/null)" || {
        rm -f "${object}"
        echo "FAIL (could not decode relocatable full-debug FFI DWARF)"
        return 1
    }
    if ! grep -q 'DW_TAG_compile_unit' <<<"${dwarf}"; then
        rm -f "${object}"
        echo "FAIL (relocatable full-debug FFI object has no compile-unit DWARF)"
        return 1
    fi
    actual_undefined="$(
        "${elf_reader}" -W -s "${object}" 2>/dev/null \
            | awk '$7 == "UND" && $8 != "" { print $8 }' \
            | grep -vE '^(\.nv\.reservedSmem\..*|__UDT(_.*)?|__UFT(_.*)?)$' \
            | LC_ALL=C sort -u
    )"
    rm -f "${object}"
    if [[ "${actual_undefined}" != "${expected_undefined}" ]]; then
        local expected_csv actual_csv
        expected_csv="$(tr '\n' ',' <<<"${expected_undefined}" | sed 's/,$//')"
        actual_csv="$(tr '\n' ',' <<<"${actual_undefined}" | sed 's/,$//')"
        echo "FAIL (full-debug FFI undefined-symbol interface drifted for ${ex}; expected=[${expected_csv}]; actual=[${actual_csv}])"
        return 1
    fi
    echo "PASS (full-debug FFI PTX assembled relocatably for ${arch}; DWARF and undefined interface verified)"
}

# The one configured cubin project cannot be judged through the root PTX
# convention. Verify its complete versioned sidecar handshake and consume the
# linked cubin's DWARF so a silently ignored interop debug policy cannot pass.
full_debug_cubin_verdict() {
    local stem="$1"
    local llvm_ir="${stem}.ll"
    local target_file="${stem}.target"
    local options_file="${stem}.options"
    local cubin="${stem}.cubin"
    local identity="${stem}.cubin.identity"
    local artifact elf_reader sections dwarf

    for artifact in "${llvm_ir}" "${target_file}" "${options_file}" "${cubin}" "${identity}"; do
        if [[ ! -s "${artifact}" ]]; then
            echo "FAIL (configured full-debug cubin artifact missing at ${artifact})"
            return 1
        fi
    done
    if [[ "$(sed -n '1p' "${target_file}")" != "${LTOIR_MODERN_ARCH}" ]] \
        || ! grep -qx 'compile-options=v1' "${target_file}"; then
        echo "FAIL (configured cubin target/options handshake is malformed)"
        return 1
    fi
    if [[ "$(sed -n '1p' "${options_file}")" != "cuda-oxide-compile-options-v2" ]] \
        || [[ "$(grep -c '^debug=full$' "${options_file}")" -ne 1 ]] \
        || [[ "$(grep -c '^debug=' "${options_file}")" -ne 1 ]]; then
        echo "FAIL (configured cubin did not record exact debug=full policy)"
        return 1
    fi
    if ! grep -q 'emissionKind: FullDebug' "${llvm_ir}"; then
        echo "FAIL (configured cubin NVVM IR has no FullDebug compile unit)"
        return 1
    fi

    elf_reader="$(command -v readelf || command -v llvm-readelf || true)"
    if [[ -z "${elf_reader}" ]]; then
        echo "FAIL (no readelf/llvm-readelf available to inspect configured cubin DWARF)"
        return 1
    fi
    sections="$("${elf_reader}" -W -S "${cubin}" 2>/dev/null)" || {
        echo "FAIL (could not read configured cubin sections)"
        return 1
    }
    if ! grep -q '[.]debug_info' <<<"${sections}" \
        || ! grep -q '[.]debug_line' <<<"${sections}"; then
        echo "FAIL (configured cubin is missing .debug_info/.debug_line)"
        return 1
    fi
    dwarf="$("${elf_reader}" --debug-dump=info "${cubin}" 2>/dev/null)" || {
        echo "FAIL (could not decode configured cubin DWARF)"
        return 1
    }
    if ! grep -q 'DW_TAG_compile_unit' <<<"${dwarf}" \
        || ! grep -q 'DW_TAG_subprogram' <<<"${dwarf}" \
        || ! grep -q 'scale_offset_f32' <<<"${dwarf}"; then
        echo "FAIL (configured cubin has no compile-unit/scale_offset_f32 DWARF)"
        return 1
    fi
    echo "PASS (configured cubin recorded debug=full and contains consumable DWARF)"
}

# ---- Runner --------------------------------------------------------------

# Build every example with full debug. Most use direct LLVM; a configured cubin
# keeps its declared modern libNVVM route. Category-specific architecture floors
# still apply even though the census launches no kernels.
run_full_debug_build() {
    local ex="$1" log="$2" cat="$3"
    local arch=""
    local debug_info_check="crates/rustc-codegen-cuda/examples/${ex}/verify-debug-info.sh"
    local invariant_shape_check="crates/rustc-codegen-cuda/examples/${ex}/verify-code-shape.sh"
    local -a args=("build" "${ex}" "--device-debug")

    case "${cat}" in
        tcgen05)          arch="sm_100a" ;;
        wgmma)            arch="sm_90a" ;;
        blackwell-mma)    arch="sm_120a" ;;
        ltoir)            arch="${LTOIR_ARCH}" ;;
        ltoir-modern)     arch="${LTOIR_MODERN_ARCH}" ;;
        auto-nvvm)        arch="${LTOIR_ARCH}" ;;
        iket)             arch="sm_90" ;;
        blackwell-compile) arch="sm_120a" ;;
        sm100-compile)    arch="sm_100a" ;;
    esac

    # Retain the instruction floors from the compile-only verifier matrix,
    # but use them with the direct LLVM build rather than emit-ltoir.
    if [[ -z "${arch}" ]] && verify_nvvm_in_compile_only "${ex}"; then
        arch="$(nvvm_verify_arch "${ex}")"
    fi
    case "${ex}" in
        cluster) arch="sm_90" ;;
        # Full debug is unsupported by the legacy LLVM 7 NVVM dialect. Keep
        # the project's configured cubin path, but use the supported modern
        # dialect at the host target or its GPU-less sm_100 floor.
        interop_cubin_identity) arch="${LTOIR_MODERN_ARCH}" ;;
    esac
    if [[ -n "${arch}" ]]; then
        args+=("--arch=${arch}")
    fi

    if [[ ${VERBOSE} -eq 1 ]]; then
        cargo oxide "${args[@]}" 2>&1 | tee "${log}"
        CARGO_EC=${PIPESTATUS[0]}
    else
        cargo oxide "${args[@]}" >"${log}" 2>&1
        CARGO_EC=$?
    fi

    # These scripts are the permanent structural contracts for full-debug
    # metadata.  They deliberately remain in this lane while the optimized
    # verify-code-shape.sh contracts below do not.
    if [[ ${CARGO_EC} -eq 0 && -f "${debug_info_check}" ]]; then
        if ! bash "${debug_info_check}" >>"${log}" 2>&1; then
            printf '%s failed its verify-debug-info.sh assertions\n' "${ex}" >>"${log}"
            CARGO_EC=1
        fi
    fi
    # `debug` is the one shape contract explicitly documented as invariant
    # across MIR optimization levels: it rejects macro-attribute source lines
    # on generated kernel-entry instructions. Other verify-code-shape scripts
    # describe optimized output and remain outside this lane.
    if [[ ${CARGO_EC} -eq 0 && "${ex}" == "debug" ]]; then
        if ! bash "${invariant_shape_check}" >>"${log}" 2>&1; then
            printf '%s failed its full-debug-invariant shape assertions\n' "${ex}" >>"${log}"
            CARGO_EC=1
        fi
    fi
}

# Run cargo oxide for ${ex} in category ${cat}. Writes to ${log}. Returns
# the cargo process exit code via the global ${CARGO_EC}.
run_cargo() {
    local ex="$1" log="$2" cat="$3"
    if [[ ${FULL_DEBUG} -eq 1 ]]; then
        run_full_debug_build "${ex}" "${log}" "${cat}"
        return
    fi
    # Rust `char` must remain plain i32 in both NVVM dialects. A runtime
    # round-trip cannot distinguish it from an incorrect zeroext i32 ABI.
    if [[ ${COMPILE_ONLY} -eq 1 && "${ex}" == "device_ffi_test" ]]; then
        local shape_check="crates/rustc-codegen-cuda/examples/${ex}/verify-code-shape.sh"
        local arch
        for arch in sm_90 sm_100; do
            local -a char_abi_args=("build" "${ex}" "--emit-nvvm-ir" "--arch=${arch}")
            if [[ ${VERBOSE} -eq 1 ]]; then
                cargo oxide "${char_abi_args[@]}" 2>&1 | tee -a "${log}"
                CARGO_EC=${PIPESTATUS[0]}
            else
                cargo oxide "${char_abi_args[@]}" >>"${log}" 2>&1
                CARGO_EC=$?
            fi
            if [[ ${CARGO_EC} -ne 0 ]]; then
                return
            fi
            local target_file="crates/rustc-codegen-cuda/examples/${ex}/${ex}.target"
            if ! grep -qx "${arch}" "${target_file}"; then
                printf '%s expected target %s in %s\n' "${ex}" "${arch}" "${target_file}" >>"${log}"
                CARGO_EC=1
                return
            fi
            if ! bash "${shape_check}" >>"${log}" 2>&1; then
                printf '%s failed its char ABI shape assertions for %s\n' "${ex}" "${arch}" >>"${log}"
                CARGO_EC=1
                return
            fi
        done
        return
    fi
    # This exact-target batch must pass both compiler routes. The second build
    # may replace the first artifact, so preserve both exit codes in one gate.
    if [[ "${cat}" == "blackwell-compile" ]]; then
        local -a llvm_args=("build" "${ex}" "--arch=sm_120a")
        local -a nvvm_args=("emit-ltoir" "${ex}" "--arch=sm_120a")
        local llvm_ec
        if [[ ${VERBOSE} -eq 1 ]]; then
            cargo oxide "${llvm_args[@]}" 2>&1 | tee "${log}"
            llvm_ec=${PIPESTATUS[0]}
        else
            cargo oxide "${llvm_args[@]}" >"${log}" 2>&1
            llvm_ec=$?
        fi
        if [[ ${llvm_ec} -ne 0 ]]; then
            CARGO_EC=${llvm_ec}
            return
        fi
        local llvm_ptx="crates/rustc-codegen-cuda/examples/${ex}/${ex}.ptx"
        local instruction_re='mma\.sp::ordered_metadata\.sync\.aligned\.m16n8k64\.row\.col\.kind::f8f6f4\.f32\.[[:alnum:]]+\.[[:alnum:]]+\.f32'
        local sparse_f16_instruction_re='mma\.sp::ordered_metadata\.sync\.aligned\.m16n8k64\.row\.col\.kind::f8f6f4\.f16\.(e2m1|e2m3|e3m2|e4m3|e5m2)\.(e2m1|e2m3|e3m2|e4m3|e5m2)\.f16'
        local dense_f32_instruction_re='mma\.sync\.aligned\.m16n8k32\.row\.col\.kind::f8f6f4\.f32\.(e2m1|e2m3|e3m2|e4m3|e5m2)\.(e2m1|e2m3|e3m2|e4m3|e5m2)\.f32'
        local dense_f16_instruction_re='mma\.sync\.aligned\.m16n8k32\.row\.col\.kind::f8f6f4\.f16\.(e2m1|e2m3|e3m2|e4m3|e5m2)\.(e2m1|e2m3|e3m2|e4m3|e5m2)\.f16'
        local standard_fp8_f32_instruction_re='mma\.sync\.aligned\.m16n8k(16|32)\.row\.col\.f32\.(e4m3|e5m2)\.(e4m3|e5m2)\.f32'
        local standard_fp8_f16_instruction_re='mma\.sync\.aligned\.m16n8k(16|32)\.row\.col\.f16\.(e4m3|e5m2)\.(e4m3|e5m2)\.f16'
        local unresolved_dense_f32_re='llvm\.nvvm\.mma\.m16n8k32\.row\.col\.kind\.f8f6f4\.f32\.'
        local unresolved_dense_f16_re='llvm\.nvvm\.mma\.m16n8k32\.row\.col\.kind\.f8f6f4\.f16\.'
        local unresolved_sparse_f16_re='llvm\.nvvm\.mma\.sp\.ordered\.metadata\.m16n8k64\.row\.col\.kind\.f8f6f4\.f16\.'
        local unresolved_standard_fp8_re='llvm\.nvvm\.mma\.m16n8k(16|32)\.row\.col\.f(16|32)\.(e4m3|e5m2)\.'
        local fp8_re='cvt\.rn\.satfinite(\.relu)?\.(e4m3x2|e5m2x2)\.f32'
        local tf32_re='cvt\.(rna|rn|rz)(\.relu)?(\.satfinite)?\.tf32\.f32'
        local ldmatrix_re='ldmatrix\.sync\.aligned\.(m16n16\.x(1|2)\.trans\.shared\.(b8|b8x16\.(b4x16_p64|b6x16_p32))|m8n16\.x(1|2|4)\.shared\.b8x16\.(b4x16_p64|b6x16_p32))'
        local instruction_count unique_instruction_count
        local sparse_f16_instruction_count unique_sparse_f16_instruction_count
        local dense_f32_instruction_count unique_dense_f32_instruction_count
        local dense_f16_instruction_count unique_dense_f16_instruction_count
        local standard_fp8_f32_instruction_count unique_standard_fp8_f32_instruction_count
        local standard_fp8_f16_instruction_count unique_standard_fp8_f16_instruction_count
        local fp8_count unique_fp8_count
        local tf32_count unique_tf32_count
        local ldmatrix_count unique_ldmatrix_count
        instruction_count="$(grep -oE "${instruction_re}" "${llvm_ptx}" 2>/dev/null | wc -l)"
        unique_instruction_count="$(grep -oE "${instruction_re}" "${llvm_ptx}" 2>/dev/null | sort -u | wc -l)"
        sparse_f16_instruction_count="$(grep -oE "${sparse_f16_instruction_re}" "${llvm_ptx}" 2>/dev/null | wc -l)"
        unique_sparse_f16_instruction_count="$(grep -oE "${sparse_f16_instruction_re}" "${llvm_ptx}" 2>/dev/null | sort -u | wc -l)"
        dense_f32_instruction_count="$(grep -oE "${dense_f32_instruction_re}" "${llvm_ptx}" 2>/dev/null | wc -l)"
        unique_dense_f32_instruction_count="$(grep -oE "${dense_f32_instruction_re}" "${llvm_ptx}" 2>/dev/null | sort -u | wc -l)"
        dense_f16_instruction_count="$(grep -oE "${dense_f16_instruction_re}" "${llvm_ptx}" 2>/dev/null | wc -l)"
        unique_dense_f16_instruction_count="$(grep -oE "${dense_f16_instruction_re}" "${llvm_ptx}" 2>/dev/null | sort -u | wc -l)"
        standard_fp8_f32_instruction_count="$(grep -oE "${standard_fp8_f32_instruction_re}" "${llvm_ptx}" 2>/dev/null | wc -l)"
        unique_standard_fp8_f32_instruction_count="$(grep -oE "${standard_fp8_f32_instruction_re}" "${llvm_ptx}" 2>/dev/null | sort -u | wc -l)"
        standard_fp8_f16_instruction_count="$(grep -oE "${standard_fp8_f16_instruction_re}" "${llvm_ptx}" 2>/dev/null | wc -l)"
        unique_standard_fp8_f16_instruction_count="$(grep -oE "${standard_fp8_f16_instruction_re}" "${llvm_ptx}" 2>/dev/null | sort -u | wc -l)"
        fp8_count="$(grep -oE "${fp8_re}" "${llvm_ptx}" 2>/dev/null | wc -l)"
        unique_fp8_count="$(grep -oE "${fp8_re}" "${llvm_ptx}" 2>/dev/null | sort -u | wc -l)"
        tf32_count="$(grep -oE "${tf32_re}" "${llvm_ptx}" 2>/dev/null | wc -l)"
        unique_tf32_count="$(grep -oE "${tf32_re}" "${llvm_ptx}" 2>/dev/null | sort -u | wc -l)"
        ldmatrix_count="$(grep -oE "${ldmatrix_re}" "${llvm_ptx}" 2>/dev/null | wc -l)"
        unique_ldmatrix_count="$(grep -oE "${ldmatrix_re}" "${llvm_ptx}" 2>/dev/null | sort -u | wc -l)"
        if [[ ! -s "${llvm_ptx}" ]] \
            || ! grep -qx '\.version 8\.7' "${llvm_ptx}" \
            || ! grep -qx '\.target sm_120a' "${llvm_ptx}" \
            || [[ ${instruction_count} -ne 25 || ${unique_instruction_count} -ne 25 ]] \
            || [[ ${sparse_f16_instruction_count} -ne 25 || ${unique_sparse_f16_instruction_count} -ne 25 ]] \
            || [[ ${dense_f32_instruction_count} -ne 25 || ${unique_dense_f32_instruction_count} -ne 25 ]] \
            || [[ ${dense_f16_instruction_count} -ne 25 || ${unique_dense_f16_instruction_count} -ne 25 ]] \
            || [[ ${standard_fp8_f32_instruction_count} -ne 8 || ${unique_standard_fp8_f32_instruction_count} -ne 8 ]] \
            || [[ ${standard_fp8_f16_instruction_count} -ne 8 || ${unique_standard_fp8_f16_instruction_count} -ne 8 ]] \
            || [[ ${fp8_count} -ne 4 || ${unique_fp8_count} -ne 4 ]] \
            || [[ ${tf32_count} -ne 10 || ${unique_tf32_count} -ne 10 ]] \
            || [[ ${ldmatrix_count} -ne 12 || ${unique_ldmatrix_count} -ne 12 ]] \
            || grep -qE "${unresolved_sparse_f16_re}|${unresolved_dense_f32_re}|${unresolved_dense_f16_re}|${unresolved_standard_fp8_re}" "${llvm_ptx}"; then
            printf 'direct LLVM route did not emit the expected sparse, dense, standard FP8, conversion, and ldmatrix instructions\n' >>"${log}"
            if [[ ${VERBOSE} -eq 1 ]]; then
                printf 'direct LLVM route did not emit the expected sparse, dense, standard FP8, conversion, and ldmatrix instructions\n'
            fi
            CARGO_EC=1
            return
        fi
        if [[ ${VERBOSE} -eq 1 ]]; then
            cargo oxide "${nvvm_args[@]}" 2>&1 | tee -a "${log}"
            CARGO_EC=${PIPESTATUS[0]}
        else
            cargo oxide "${nvvm_args[@]}" >>"${log}" 2>&1
            CARGO_EC=$?
        fi
        if [[ ${CARGO_EC} -ne 0 ]]; then
            return
        fi
        local nvvm_ll="crates/rustc-codegen-cuda/examples/${ex}/${ex}.ll"
        local inline_fp8_re='call i16 asm "cvt\.rn\.satfinite(\.relu)?\.(e4m3x2|e5m2x2)\.f32 \$0, \$2, \$1;", "=h,f,f"'
        local inline_tf32_re='call i32 asm "cvt\.(rna|rn|rz)(\.relu)?(\.satfinite)?\.tf32\.f32 \$0, \$1;", "=r,f"'
        local inline_fp8_count unique_inline_fp8_count
        local inline_tf32_count unique_inline_tf32_count
        local inline_ldmatrix_count unique_inline_ldmatrix_count
        local inline_sparse_f16_count unique_inline_sparse_f16_count
        local inline_dense_f32_count unique_inline_dense_f32_count
        local inline_dense_f16_count unique_inline_dense_f16_count
        local inline_standard_fp8_f32_count unique_inline_standard_fp8_f32_count
        local inline_standard_fp8_f16_count unique_inline_standard_fp8_f16_count
        inline_fp8_count="$(grep -oE "${inline_fp8_re}" "${nvvm_ll}" 2>/dev/null | wc -l)"
        unique_inline_fp8_count="$(grep -oE "${inline_fp8_re}" "${nvvm_ll}" 2>/dev/null | sort -u | wc -l)"
        inline_tf32_count="$(grep -oE "${inline_tf32_re}" "${nvvm_ll}" 2>/dev/null | wc -l)"
        unique_inline_tf32_count="$(grep -oE "${inline_tf32_re}" "${nvvm_ll}" 2>/dev/null | sort -u | wc -l)"
        inline_ldmatrix_count="$(grep -oE "${ldmatrix_re}" "${nvvm_ll}" 2>/dev/null | wc -l)"
        unique_inline_ldmatrix_count="$(grep -oE "${ldmatrix_re}" "${nvvm_ll}" 2>/dev/null | sort -u | wc -l)"
        inline_sparse_f16_count="$(grep -oE "${sparse_f16_instruction_re}" "${nvvm_ll}" 2>/dev/null | wc -l)"
        unique_inline_sparse_f16_count="$(grep -oE "${sparse_f16_instruction_re}" "${nvvm_ll}" 2>/dev/null | sort -u | wc -l)"
        inline_dense_f32_count="$(grep -oE "${dense_f32_instruction_re}" "${nvvm_ll}" 2>/dev/null | wc -l)"
        unique_inline_dense_f32_count="$(grep -oE "${dense_f32_instruction_re}" "${nvvm_ll}" 2>/dev/null | sort -u | wc -l)"
        inline_dense_f16_count="$(grep -oE "${dense_f16_instruction_re}" "${nvvm_ll}" 2>/dev/null | wc -l)"
        unique_inline_dense_f16_count="$(grep -oE "${dense_f16_instruction_re}" "${nvvm_ll}" 2>/dev/null | sort -u | wc -l)"
        inline_standard_fp8_f32_count="$(grep -oE "${standard_fp8_f32_instruction_re}" "${nvvm_ll}" 2>/dev/null | wc -l)"
        unique_inline_standard_fp8_f32_count="$(grep -oE "${standard_fp8_f32_instruction_re}" "${nvvm_ll}" 2>/dev/null | sort -u | wc -l)"
        inline_standard_fp8_f16_count="$(grep -oE "${standard_fp8_f16_instruction_re}" "${nvvm_ll}" 2>/dev/null | wc -l)"
        unique_inline_standard_fp8_f16_count="$(grep -oE "${standard_fp8_f16_instruction_re}" "${nvvm_ll}" 2>/dev/null | sort -u | wc -l)"
        if [[ ! -s "${nvvm_ll}" ]] \
            || [[ ${inline_fp8_count} -ne 4 || ${unique_inline_fp8_count} -ne 4 ]] \
            || [[ ${inline_tf32_count} -ne 10 || ${unique_inline_tf32_count} -ne 10 ]] \
            || [[ ${inline_ldmatrix_count} -ne 12 || ${unique_inline_ldmatrix_count} -ne 12 ]] \
            || [[ ${inline_sparse_f16_count} -ne 25 || ${unique_inline_sparse_f16_count} -ne 25 ]] \
            || [[ ${inline_dense_f32_count} -ne 25 || ${unique_inline_dense_f32_count} -ne 25 ]] \
            || [[ ${inline_dense_f16_count} -ne 25 || ${unique_inline_dense_f16_count} -ne 25 ]] \
            || [[ ${inline_standard_fp8_f32_count} -ne 8 || ${unique_inline_standard_fp8_f32_count} -ne 8 ]] \
            || [[ ${inline_standard_fp8_f16_count} -ne 8 || ${unique_inline_standard_fp8_f16_count} -ne 8 ]] \
            || grep -qE 'llvm\.nvvm\.(ff\.to\.|f2tf32|ldmatrix\.)' "${nvvm_ll}" \
            || grep -qE "${unresolved_sparse_f16_re}|${unresolved_dense_f32_re}|${unresolved_dense_f16_re}|${unresolved_standard_fp8_re}" "${nvvm_ll}"; then
            printf 'libNVVM route did not emit the expected sparse, dense, standard FP8, conversion, and ldmatrix inline-PTX calls\n' >>"${log}"
            if [[ ${VERBOSE} -eq 1 ]]; then
                printf 'libNVVM route did not emit the expected sparse, dense, standard FP8, conversion, and ldmatrix inline-PTX calls\n'
            fi
            CARGO_EC=1
        fi
        return
    fi

    # The f32 redux family admits only Blackwell family 10x in the generated
    # target gating, so this batch pins sm_100a.
    if [[ "${cat}" == "sm100-compile" ]]; then
        local -a llvm_args=("build" "${ex}" "--arch=sm_100a")
        local llvm_ec
        if [[ ${VERBOSE} -eq 1 ]]; then
            cargo oxide "${llvm_args[@]}" 2>&1 | tee "${log}"
            llvm_ec=${PIPESTATUS[0]}
        else
            cargo oxide "${llvm_args[@]}" >"${log}" 2>&1
            llvm_ec=$?
        fi
        CARGO_EC=${llvm_ec}
        if [[ ${llvm_ec} -ne 0 ]]; then
            return
        fi
        local llvm_ptx="crates/rustc-codegen-cuda/examples/${ex}/${ex}.ptx"
        local redux_f32_re='redux\.sync\.(min|max)(\.abs)?(\.NaN)?\.f32'
        # Each kernel must emit all eight forms inline. Count per entry body:
        # the shared #[device] helper also survives as a standalone .func copy
        # the kernels never call, and an aggregate count could go green with
        # the forms split unevenly across the two kernels.
        local kernel body kernels_ok=1
        for kernel in redux_f32_finite redux_f32_nan_policy; do
            body="$(awk "/^\\.visible \\.entry ${kernel}\\(/,/^}/" "${llvm_ptx}" 2>/dev/null)"
            if [[ "$(grep -oE "${redux_f32_re}" <<<"${body}" | wc -l)" -ne 8 ]] \
                || [[ "$(grep -oE "${redux_f32_re}" <<<"${body}" | sort -u | wc -l)" -ne 8 ]]; then
                kernels_ok=0
            fi
        done
        if [[ ! -s "${llvm_ptx}" ]] \
            || ! grep -qx '\.version 8\.6' "${llvm_ptx}" \
            || ! grep -qx '\.target sm_100a' "${llvm_ptx}" \
            || [[ ${kernels_ok} -ne 1 ]]; then
            printf 'direct LLVM route did not emit the expected redux.sync f32 instructions\n' >>"${log}"
            if [[ ${VERBOSE} -eq 1 ]]; then
                printf 'direct LLVM route did not emit the expected redux.sync f32 instructions\n'
            fi
            CARGO_EC=1
        fi
        return
    fi

    # The generated tcgen05 families must pass both compiler routes.
    if [[ ${COMPILE_ONLY} -eq 1 && "${ex}" == "tcgen05" ]]; then
        local cp_re='tcgen05\.cp\.cta_group::[12]\.(128x128b|128x256b|32x128b\.warpx4|4x256b|64x128b\.warpx2::(01_23|02_13))(\.b8x16\.(b4x16_p64|b6x16_p32))?[[:space:]]'
        local mma_base_re='tcgen05\.mma(\.sp)?\.cta_group::[12]\.kind::(f16|tf32|f8f6f4|i8)\.collector::a::(discard|lastuse|fill|use)(\.ashift)?'
        local mma_base_plain_re='tcgen05\.mma\.cta_group::[12]\.kind::(f16|tf32|f8f6f4|i8)\.collector::a::(discard|lastuse|fill|use)(\.ashift)?'
        local mma_base_sp_re='tcgen05\.mma\.sp\.cta_group::[12]\.kind::(f16|tf32|f8f6f4|i8)\.collector::a::(discard|lastuse|fill|use)(\.ashift)?'
        local mma_ws_re='tcgen05\.mma\.ws(\.sp)?\.cta_group::[12]\.kind::(f16|tf32|f8f6f4|i8)\.collector::b[0-3]::(discard|lastuse|fill|use)'
        local mma_ws_plain_re='tcgen05\.mma\.ws\.cta_group::[12]\.kind::(f16|tf32|f8f6f4|i8)\.collector::b[0-3]::(discard|lastuse|fill|use)'
        local mma_ws_sp_re='tcgen05\.mma\.ws\.sp\.cta_group::[12]\.kind::(f16|tf32|f8f6f4|i8)\.collector::b[0-3]::(discard|lastuse|fill|use)'
        local ptx_shared_a_re='\[[^]]+\],[[:space:]]+%rd[0-9]+'
        local ptx_tensor_a_re='\[[^]]+\],[[:space:]]+\[%r[0-9]+\]'
        local ptx_zero_mask_re='%enable_pred,[[:space:]]+%rd[0-9]+;'
        local nvvm_shared_a_re='\[\$0\],[[:space:]]+\$1'
        local nvvm_tensor_a_re='\[\$0\],[[:space:]]+\[\$1\]'
        local nvvm_zero_mask_re='%enable_pred,[[:space:]]+\$[56];'
        local unresolved_mma_re='llvm\.nvvm\.tcgen05\.mma'
        local ld_re='tcgen05\.ld\.sync\.aligned\.(16x64b|16x128b|16x256b|32x32b)\.x(1|2|4|8|16|32|64|128)(\.pack::16b)?\.b32'
        local ld_pack16_re='tcgen05\.ld\.sync\.aligned\.(16x64b|16x128b|16x256b|32x32b)\.x(1|2|4|8|16|32|64|128)\.pack::16b\.b32'
        local ld_raw_re='tcgen05\.ld\.sync\.aligned\.(16x64b|16x128b|16x256b|32x32b)\.x(1|2|4|8|16|32|64|128)\.b32'
        local st_re='tcgen05\.st\.sync\.aligned\.(16x64b|16x128b|16x256b|32x32b)\.x(1|2|4|8|16|32|64|128)(\.unpack::16b)?\.b32'
        local st_unpack16_re='tcgen05\.st\.sync\.aligned\.(16x64b|16x128b|16x256b|32x32b)\.x(1|2|4|8|16|32|64|128)\.unpack::16b\.b32'
        local st_raw_re='tcgen05\.st\.sync\.aligned\.(16x64b|16x128b|16x256b|32x32b)\.x(1|2|4|8|16|32|64|128)\.b32'
        local ld_offset_re='tcgen05\.ld\.sync\.aligned\.16x32bx2\.x(1|2|4|8|16|32|64|128)(\.pack::16b)?\.b32'
        local ld_offset_pack16_re='tcgen05\.ld\.sync\.aligned\.16x32bx2\.x(1|2|4|8|16|32|64|128)\.pack::16b\.b32'
        local ld_offset_raw_re='tcgen05\.ld\.sync\.aligned\.16x32bx2\.x(1|2|4|8|16|32|64|128)\.b32'
        local st_offset_re='tcgen05\.st\.sync\.aligned\.16x32bx2\.x(1|2|4|8|16|32|64|128)(\.unpack::16b)?\.b32'
        local st_offset_unpack16_re='tcgen05\.st\.sync\.aligned\.16x32bx2\.x(1|2|4|8|16|32|64|128)\.unpack::16b\.b32'
        local st_offset_raw_re='tcgen05\.st\.sync\.aligned\.16x32bx2\.x(1|2|4|8|16|32|64|128)\.b32'
        local commit_multicast_cg1_re='tcgen05\.commit\.cta_group::1\.mbarrier::arrive::one\.shared::cluster\.multicast::cluster\.b64'
        local shift_down_cg1_re='tcgen05\.shift\.cta_group::1\.down'
        local shift_down_cg2_re='tcgen05\.shift\.cta_group::2\.down'
        local unresolved_control_re='llvm\.nvvm\.tcgen05\.(commit\.mc\.shared\.cg1|shift\.down\.cg[12])'
        local llvm_ptx="crates/rustc-codegen-cuda/examples/${ex}/${ex}.ptx"
        local -a llvm_args=("build" "${ex}" "--arch=sm_100a")
        local llvm_ec
        if [[ ${VERBOSE} -eq 1 ]]; then
            cargo oxide "${llvm_args[@]}" 2>&1 | tee "${log}"
            llvm_ec=${PIPESTATUS[0]}
        else
            cargo oxide "${llvm_args[@]}" >"${log}" 2>&1
            llvm_ec=$?
        fi
        if [[ ${llvm_ec} -ne 0 ]]; then
            CARGO_EC=${llvm_ec}
            return
        fi

        local llvm_cg1 llvm_cg2 llvm_cp_count llvm_cp_unique
        llvm_cg1="$(awk '/^\.visible \.entry compile_tcgen05_cp_cg1\(/,/^}/' "${llvm_ptx}" 2>/dev/null)"
        llvm_cg2="$(awk '/^\.visible \.entry compile_tcgen05_cp_cg2\(/,/^}/' "${llvm_ptx}" 2>/dev/null)"
        llvm_cp_count="$(grep -oE "${cp_re}" <<<"${llvm_cg1}"$'\n'"${llvm_cg2}" | wc -l)"
        llvm_cp_unique="$(grep -oE "${cp_re}" <<<"${llvm_cg1}"$'\n'"${llvm_cg2}" | sort -u | wc -l)"
        if [[ -z "${llvm_cg1}" || -z "${llvm_cg2}" ]] \
            || ! grep -qx '\.target sm_100a' "${llvm_ptx}" \
            || [[ $(grep -oE "${cp_re}" <<<"${llvm_cg1}" | wc -l) -ne 18 ]] \
            || [[ $(grep -oE "${cp_re}" <<<"${llvm_cg2}" | wc -l) -ne 18 ]] \
            || grep -q 'tcgen05\.cp\.cta_group::2\.' <<<"${llvm_cg1}" \
            || grep -q 'tcgen05\.cp\.cta_group::1\.' <<<"${llvm_cg2}" \
            || [[ ${llvm_cp_count} -ne 36 || ${llvm_cp_unique} -ne 36 ]]; then
            printf 'tcgen05 LLVM route expected 18 copy forms per CTA group and 36 unique total; got %s/%s total/unique\n' \
                "${llvm_cp_count}" "${llvm_cp_unique}" >>"${log}"
            CARGO_EC=1
            return
        fi

        local llvm_control_cg1 llvm_control_cg2
        llvm_control_cg1="$(awk '/^\.visible \.entry compile_tcgen05_control_cg1\(/,/^}/' "${llvm_ptx}" 2>/dev/null)"
        llvm_control_cg2="$(awk '/^\.visible \.entry compile_tcgen05_control_cg2\(/,/^}/' "${llvm_ptx}" 2>/dev/null)"
        if [[ -z "${llvm_control_cg1}" || -z "${llvm_control_cg2}" ]] \
            || [[ $(grep -oE "${commit_multicast_cg1_re}" <<<"${llvm_control_cg1}" | wc -l) -ne 1 ]] \
            || [[ $(grep -oE "${shift_down_cg1_re}" <<<"${llvm_control_cg1}" | wc -l) -ne 1 ]] \
            || [[ $(grep -oE "${shift_down_cg2_re}" <<<"${llvm_control_cg2}" | wc -l) -ne 1 ]] \
            || grep -qE 'tcgen05\.(commit|shift)\.cta_group::2' <<<"${llvm_control_cg1}" \
            || grep -qE 'tcgen05\.(commit|shift)\.cta_group::1' <<<"${llvm_control_cg2}" \
            || grep -qE "${unresolved_control_re}" "${llvm_ptx}"; then
            printf 'tcgen05 LLVM route did not emit the exact cg1 multicast commit and cg1/cg2 shift-down forms\n' >>"${log}"
            CARGO_EC=1
            return
        fi

        # The base MMA coverage is split into cg1/cg2 kernels (ptxas enforces
        # one tcgen05 granularity per function); the aggregate counts below
        # run over both bodies concatenated, plus per-kernel purity checks.
        local llvm_mma_base_cg1 llvm_mma_base_cg2
        local llvm_mma_base llvm_mma_ws llvm_mma_base_count llvm_mma_base_unique
        local llvm_mma_ws_count llvm_mma_ws_unique
        llvm_mma_base_cg1="$(awk '/^\.visible \.entry compile_tcgen05_mma_base_cg1\(/,/^}/' "${llvm_ptx}" 2>/dev/null)"
        llvm_mma_base_cg2="$(awk '/^\.visible \.entry compile_tcgen05_mma_base_cg2\(/,/^}/' "${llvm_ptx}" 2>/dev/null)"
        llvm_mma_base="${llvm_mma_base_cg1}"$'\n'"${llvm_mma_base_cg2}"
        llvm_mma_ws="$(awk '/^\.visible \.entry compile_tcgen05_mma_ws\(/,/^}/' "${llvm_ptx}" 2>/dev/null)"
        llvm_mma_base_count="$(grep -oE "${mma_base_re}" <<<"${llvm_mma_base}" | wc -l)"
        llvm_mma_base_unique="$(grep -oE "${mma_base_re}" <<<"${llvm_mma_base}" | sort -u | wc -l)"
        llvm_mma_ws_count="$(grep -oE "${mma_ws_re}" <<<"${llvm_mma_ws}" | wc -l)"
        llvm_mma_ws_unique="$(grep -oE "${mma_ws_re}" <<<"${llvm_mma_ws}" | sort -u | wc -l)"
        if [[ -z "${llvm_mma_base_cg1}" || -z "${llvm_mma_base_cg2}" || -z "${llvm_mma_ws}" ]] \
            || grep -qE '\.cta_group::2\.' <<<"${llvm_mma_base_cg1}" \
            || grep -qE '\.cta_group::1\.' <<<"${llvm_mma_base_cg2}" \
            || [[ ${llvm_mma_base_count} -ne 9 || ${llvm_mma_base_unique} -ne 8 ]] \
            || [[ ${llvm_mma_ws_count} -ne 16 || ${llvm_mma_ws_unique} -ne 10 ]] \
            || [[ $(grep -oE "${mma_base_plain_re}" <<<"${llvm_mma_base}" | wc -l) -ne 6 ]] \
            || [[ $(grep -oE "${mma_base_sp_re}" <<<"${llvm_mma_base}" | wc -l) -ne 3 ]] \
            || [[ $(grep -oE "${mma_ws_plain_re}" <<<"${llvm_mma_ws}" | wc -l) -ne 12 ]] \
            || [[ $(grep -oE "${mma_ws_sp_re}" <<<"${llvm_mma_ws}" | wc -l) -ne 4 ]] \
            || [[ $(grep -cE "${mma_base_plain_re}[[:space:]]+${ptx_shared_a_re}" <<<"${llvm_mma_base}") -ne 4 ]] \
            || [[ $(grep -cE "${mma_base_plain_re}[[:space:]]+${ptx_tensor_a_re}" <<<"${llvm_mma_base}") -ne 2 ]] \
            || [[ $(grep -cE "${mma_base_sp_re}[[:space:]]+${ptx_shared_a_re}" <<<"${llvm_mma_base}") -ne 1 ]] \
            || [[ $(grep -cE "${mma_base_sp_re}[[:space:]]+${ptx_tensor_a_re}" <<<"${llvm_mma_base}") -ne 2 ]] \
            || [[ $(grep -cE "${mma_ws_plain_re}[[:space:]]+${ptx_shared_a_re}" <<<"${llvm_mma_ws}") -ne 2 ]] \
            || [[ $(grep -cE "${mma_ws_plain_re}[[:space:]]+${ptx_tensor_a_re}" <<<"${llvm_mma_ws}") -ne 10 ]] \
            || [[ $(grep -cE "${mma_ws_sp_re}[[:space:]]+${ptx_shared_a_re}" <<<"${llvm_mma_ws}") -ne 2 ]] \
            || [[ $(grep -cE "${mma_ws_sp_re}[[:space:]]+${ptx_tensor_a_re}" <<<"${llvm_mma_ws}") -ne 2 ]] \
            || [[ $(grep -oE '\.cta_group::1\.' <<<"${llvm_mma_base}" | wc -l) -ne 5 ]] \
            || [[ $(grep -oE '\.cta_group::2\.' <<<"${llvm_mma_base}" | wc -l) -ne 4 ]] \
            || [[ $(grep -oE '\.cta_group::1\.' <<<"${llvm_mma_ws}" | wc -l) -ne 16 ]] \
            || grep -qE '\.cta_group::2\.' <<<"${llvm_mma_ws}" \
            || [[ $(grep -oE '\.kind::f16\.' <<<"${llvm_mma_base}" | wc -l) -ne 3 ]] \
            || [[ $(grep -oE '\.kind::tf32\.' <<<"${llvm_mma_base}" | wc -l) -ne 2 ]] \
            || [[ $(grep -oE '\.kind::f8f6f4\.' <<<"${llvm_mma_base}" | wc -l) -ne 2 ]] \
            || [[ $(grep -oE '\.kind::i8\.' <<<"${llvm_mma_base}" | wc -l) -ne 2 ]] \
            || [[ $(grep -oE '\.kind::f16\.' <<<"${llvm_mma_ws}" | wc -l) -ne 3 ]] \
            || [[ $(grep -oE '\.kind::tf32\.' <<<"${llvm_mma_ws}" | wc -l) -ne 3 ]] \
            || [[ $(grep -oE '\.kind::f8f6f4\.' <<<"${llvm_mma_ws}" | wc -l) -ne 8 ]] \
            || [[ $(grep -oE '\.kind::i8\.' <<<"${llvm_mma_ws}" | wc -l) -ne 2 ]] \
            || [[ $(grep -oE '\.collector::a::discard' <<<"${llvm_mma_base}" | wc -l) -ne 3 ]] \
            || [[ $(grep -oE '\.collector::a::lastuse' <<<"${llvm_mma_base}" | wc -l) -ne 2 ]] \
            || [[ $(grep -oE '\.collector::a::fill' <<<"${llvm_mma_base}" | wc -l) -ne 2 ]] \
            || [[ $(grep -oE '\.collector::a::use' <<<"${llvm_mma_base}" | wc -l) -ne 2 ]] \
            || [[ $(grep -oE '\.collector::b0::discard' <<<"${llvm_mma_ws}" | wc -l) -ne 7 ]] \
            || [[ $(grep -oE '\.collector::b1::(lastuse|fill)' <<<"${llvm_mma_ws}" | wc -l) -ne 3 ]] \
            || [[ $(grep -oE '\.collector::b2::(fill|use)' <<<"${llvm_mma_ws}" | wc -l) -ne 3 ]] \
            || [[ $(grep -oE '\.collector::b3::(use|discard)' <<<"${llvm_mma_ws}" | wc -l) -ne 3 ]] \
            || [[ $(grep -oE '\.ashift' <<<"${llvm_mma_base}" | wc -l) -ne 2 ]] \
            || grep -qE '\.ashift' <<<"${llvm_mma_ws}" \
            || [[ $(grep -cE "${mma_ws_re}.*${ptx_zero_mask_re}" <<<"${llvm_mma_ws}") -ne 4 ]] \
            || [[ $(grep -cE "${mma_ws_plain_re}[[:space:]]+${ptx_shared_a_re}.*${ptx_zero_mask_re}" <<<"${llvm_mma_ws}") -ne 1 ]] \
            || [[ $(grep -cE "${mma_ws_plain_re}[[:space:]]+${ptx_tensor_a_re}.*${ptx_zero_mask_re}" <<<"${llvm_mma_ws}") -ne 1 ]] \
            || [[ $(grep -cE "${mma_ws_sp_re}[[:space:]]+${ptx_shared_a_re}.*${ptx_zero_mask_re}" <<<"${llvm_mma_ws}") -ne 1 ]] \
            || [[ $(grep -cE "${mma_ws_sp_re}[[:space:]]+${ptx_tensor_a_re}.*${ptx_zero_mask_re}" <<<"${llvm_mma_ws}") -ne 1 ]] \
            || grep -qE "${unresolved_mma_re}" "${llvm_ptx}"; then
            printf 'tcgen05 LLVM route expected 9 base and 16 warp-specialized MMA calls across all forms; got %s/%s and %s/%s total/unique\n' \
                "${llvm_mma_base_count}" "${llvm_mma_base_unique}" "${llvm_mma_ws_count}" "${llvm_mma_ws_unique}" >>"${log}"
            CARGO_EC=1
            return
        fi

        local llvm_ld llvm_ld_count llvm_ld_unique llvm_ld_pack16 llvm_ld_raw
        llvm_ld="$(awk '/^\.visible \.entry compile_tcgen05_ld\(/,/^}/' "${llvm_ptx}" 2>/dev/null)"
        llvm_ld_count="$(grep -oE "${ld_re}" <<<"${llvm_ld}" | wc -l)"
        llvm_ld_unique="$(grep -oE "${ld_re}" <<<"${llvm_ld}" | sort -u | wc -l)"
        llvm_ld_pack16="$(grep -oE "${ld_pack16_re}" <<<"${llvm_ld}" | wc -l)"
        llvm_ld_raw="$(grep -oE "${ld_raw_re}" <<<"${llvm_ld}" | wc -l)"
        if [[ -z "${llvm_ld}" ]] \
            || [[ ${llvm_ld_count} -ne 58 || ${llvm_ld_unique} -ne 58 ]] \
            || [[ ${llvm_ld_pack16} -ne 29 || ${llvm_ld_raw} -ne 29 ]] \
            || grep -q 'llvm\.nvvm\.tcgen05\.ld' "${llvm_ptx}"; then
            printf 'tcgen05 LLVM route expected 58 unique loads (29 raw and 29 pack16); got %s/%s total/unique and %s/%s raw/pack16\n' \
                "${llvm_ld_count}" "${llvm_ld_unique}" "${llvm_ld_raw}" "${llvm_ld_pack16}" >>"${log}"
            CARGO_EC=1
            return
        fi

        local llvm_st llvm_st_count llvm_st_unique llvm_st_unpack16 llvm_st_raw
        llvm_st="$(awk '/^\.visible \.entry compile_tcgen05_st\(/,/^}/' "${llvm_ptx}" 2>/dev/null)"
        llvm_st_count="$(grep -oE "${st_re}" <<<"${llvm_st}" | wc -l)"
        llvm_st_unique="$(grep -oE "${st_re}" <<<"${llvm_st}" | sort -u | wc -l)"
        llvm_st_unpack16="$(grep -oE "${st_unpack16_re}" <<<"${llvm_st}" | wc -l)"
        llvm_st_raw="$(grep -oE "${st_raw_re}" <<<"${llvm_st}" | wc -l)"
        if [[ -z "${llvm_st}" ]] \
            || [[ ${llvm_st_count} -ne 58 || ${llvm_st_unique} -ne 58 ]] \
            || [[ ${llvm_st_unpack16} -ne 29 || ${llvm_st_raw} -ne 29 ]] \
            || grep -q 'llvm\.nvvm\.tcgen05\.st' "${llvm_ptx}"; then
            printf 'tcgen05 LLVM route expected 58 unique stores (29 raw and 29 unpack16); got %s/%s total/unique and %s/%s raw/unpack16\n' \
                "${llvm_st_count}" "${llvm_st_unique}" "${llvm_st_raw}" "${llvm_st_unpack16}" >>"${log}"
            CARGO_EC=1
            return
        fi

        local llvm_ld_offset llvm_ld_offset_count llvm_ld_offset_unique llvm_ld_offset_pack16 llvm_ld_offset_raw
        llvm_ld_offset="$(awk '/^\.visible \.entry compile_tcgen05_ld_offset\(/,/^}/' "${llvm_ptx}" 2>/dev/null)"
        llvm_ld_offset_count="$(grep -oE "${ld_offset_re}" <<<"${llvm_ld_offset}" | wc -l)"
        llvm_ld_offset_unique="$(grep -oE "${ld_offset_re}" <<<"${llvm_ld_offset}" | sort -u | wc -l)"
        llvm_ld_offset_pack16="$(grep -oE "${ld_offset_pack16_re}" <<<"${llvm_ld_offset}" | wc -l)"
        llvm_ld_offset_raw="$(grep -oE "${ld_offset_raw_re}" <<<"${llvm_ld_offset}" | wc -l)"
        if [[ -z "${llvm_ld_offset}" ]] \
            || [[ ${llvm_ld_offset_count} -ne 16 || ${llvm_ld_offset_unique} -ne 16 ]] \
            || [[ ${llvm_ld_offset_pack16} -ne 8 || ${llvm_ld_offset_raw} -ne 8 ]] \
            || [[ $(grep -c ', 16;' <<<"${llvm_ld_offset}") -ne 16 ]]; then
            printf 'tcgen05 LLVM route expected 16 unique offset loads with immediate 16; got %s/%s total/unique and %s/%s raw/pack16\n' \
                "${llvm_ld_offset_count}" "${llvm_ld_offset_unique}" "${llvm_ld_offset_raw}" "${llvm_ld_offset_pack16}" >>"${log}"
            CARGO_EC=1
            return
        fi

        local llvm_st_offset llvm_st_offset_count llvm_st_offset_unique llvm_st_offset_unpack16 llvm_st_offset_raw
        llvm_st_offset="$(awk '/^\.visible \.entry compile_tcgen05_st_offset\(/,/^}/' "${llvm_ptx}" 2>/dev/null)"
        llvm_st_offset_count="$(grep -oE "${st_offset_re}" <<<"${llvm_st_offset}" | wc -l)"
        llvm_st_offset_unique="$(grep -oE "${st_offset_re}" <<<"${llvm_st_offset}" | sort -u | wc -l)"
        llvm_st_offset_unpack16="$(grep -oE "${st_offset_unpack16_re}" <<<"${llvm_st_offset}" | wc -l)"
        llvm_st_offset_raw="$(grep -oE "${st_offset_raw_re}" <<<"${llvm_st_offset}" | wc -l)"
        if [[ -z "${llvm_st_offset}" ]] \
            || [[ ${llvm_st_offset_count} -ne 16 || ${llvm_st_offset_unique} -ne 16 ]] \
            || [[ ${llvm_st_offset_unpack16} -ne 8 || ${llvm_st_offset_raw} -ne 8 ]] \
            || [[ $(grep -c ', 16,' <<<"${llvm_st_offset}") -ne 16 ]]; then
            printf 'tcgen05 LLVM route expected 16 unique offset stores with immediate 16; got %s/%s total/unique and %s/%s raw/unpack16\n' \
                "${llvm_st_offset_count}" "${llvm_st_offset_unique}" "${llvm_st_offset_raw}" "${llvm_st_offset_unpack16}" >>"${log}"
            CARGO_EC=1
            return
        fi

        local -a nvvm_args=("emit-ltoir" "${ex}" "--arch=sm_100a")
        if [[ ${VERBOSE} -eq 1 ]]; then
            cargo oxide "${nvvm_args[@]}" 2>&1 | tee -a "${log}"
            CARGO_EC=${PIPESTATUS[0]}
        else
            cargo oxide "${nvvm_args[@]}" >>"${log}" 2>&1
            CARGO_EC=$?
        fi
        if [[ ${CARGO_EC} -ne 0 ]]; then
            return
        fi

        local nvvm_ll="crates/rustc-codegen-cuda/examples/${ex}/${ex}.ll"
        local nvvm_target="crates/rustc-codegen-cuda/examples/${ex}/${ex}.target"
        local nvvm_cg1 nvvm_cg2 nvvm_cp_count nvvm_cp_unique
        nvvm_cg1="$(awk '/^define .*@compile_tcgen05_cp_cg1\(/,/^}/' "${nvvm_ll}" 2>/dev/null)"
        nvvm_cg2="$(awk '/^define .*@compile_tcgen05_cp_cg2\(/,/^}/' "${nvvm_ll}" 2>/dev/null)"
        nvvm_cp_count="$(grep -oE "${cp_re}" <<<"${nvvm_cg1}"$'\n'"${nvvm_cg2}" | wc -l)"
        nvvm_cp_unique="$(grep -oE "${cp_re}" <<<"${nvvm_cg1}"$'\n'"${nvvm_cg2}" | sort -u | wc -l)"
        if [[ -z "${nvvm_cg1}" || -z "${nvvm_cg2}" ]] \
            || [[ $(grep -oE "${cp_re}" <<<"${nvvm_cg1}" | wc -l) -ne 18 ]] \
            || [[ $(grep -oE "${cp_re}" <<<"${nvvm_cg2}" | wc -l) -ne 18 ]] \
            || grep -q 'tcgen05\.cp\.cta_group::2\.' <<<"${nvvm_cg1}" \
            || grep -q 'tcgen05\.cp\.cta_group::1\.' <<<"${nvvm_cg2}" \
            || [[ ${nvvm_cp_count} -ne 36 || ${nvvm_cp_unique} -ne 36 ]] \
            || grep -q 'llvm\.nvvm\.tcgen05\.cp' "${nvvm_ll}"; then
            printf 'tcgen05 libNVVM route expected 18 inline copy forms per CTA group and 36 unique total; got %s/%s total/unique\n' \
                "${nvvm_cp_count}" "${nvvm_cp_unique}" >>"${log}"
            CARGO_EC=1
        fi

        local nvvm_control_cg1 nvvm_control_cg2
        nvvm_control_cg1="$(awk '/^define .*@compile_tcgen05_control_cg1\(/,/^}/' "${nvvm_ll}" 2>/dev/null)"
        nvvm_control_cg2="$(awk '/^define .*@compile_tcgen05_control_cg2\(/,/^}/' "${nvvm_ll}" 2>/dev/null)"
        local -a nvvm_control_attrs=()
        local nvvm_control_attr_line
        while IFS= read -r nvvm_control_attr_line; do
            nvvm_control_attrs+=("${nvvm_control_attr_line}")
        done < <(
            sed -nE '/call void asm sideeffect "tcgen05\.(commit|shift)\.cta_group::[12]/s/.* (#[0-9]+)$/\1/p' \
                <<<"${nvvm_control_cg1}"$'\n'"${nvvm_control_cg2}"
        )
        local nvvm_control_convergent=1 control_attr control_attr_definition
        if [[ ${#nvvm_control_attrs[@]} -ne 4 ]]; then
            nvvm_control_convergent=0
        else
            for control_attr in "${nvvm_control_attrs[@]}"; do
                control_attr_definition="$(grep -E "^attributes ${control_attr} = \\{[^}]*\\}$" "${nvvm_ll}")"
                if [[ $(wc -l <<<"${control_attr_definition}") -ne 1 ]] \
                    || ! grep -qw convergent <<<"${control_attr_definition}"; then
                    nvvm_control_convergent=0
                    break
                fi
            done
        fi
        if [[ -z "${nvvm_control_cg1}" || -z "${nvvm_control_cg2}" ]] \
            || [[ $(grep -oE "${commit_multicast_cg1_re}" <<<"${nvvm_control_cg1}" | wc -l) -ne 1 ]] \
            || [[ $(grep -oE "${shift_down_cg1_re}" <<<"${nvvm_control_cg1}" | wc -l) -ne 1 ]] \
            || [[ $(grep -oE "${shift_down_cg2_re}" <<<"${nvvm_control_cg2}" | wc -l) -ne 1 ]] \
            || ! grep -q 'asm sideeffect' <<<"${nvvm_control_cg1}" \
            || ! grep -q 'asm sideeffect' <<<"${nvvm_control_cg2}" \
            || [[ ${nvvm_control_convergent} -ne 1 ]] \
            || grep -qE 'tcgen05\.(commit|shift)\.cta_group::2' <<<"${nvvm_control_cg1}" \
            || grep -qE 'tcgen05\.(commit|shift)\.cta_group::1' <<<"${nvvm_control_cg2}" \
            || grep -qE "${unresolved_control_re}" "${nvvm_ll}"; then
            printf 'tcgen05 libNVVM route did not emit the exact cg1 multicast commit and cg1/cg2 shift-down inline assembly\n' >>"${log}"
            CARGO_EC=1
        fi

        # Same cg1/cg2 split as the PTX route: aggregate over both bodies.
        local nvvm_mma_base_cg1 nvvm_mma_base_cg2
        local nvvm_mma_base nvvm_mma_ws nvvm_mma_base_count nvvm_mma_base_unique
        local nvvm_mma_ws_count nvvm_mma_ws_unique nvvm_mma_inline_count nvvm_mma_memory_count
        nvvm_mma_base_cg1="$(awk '/^define .*@compile_tcgen05_mma_base_cg1\(/,/^}/' "${nvvm_ll}" 2>/dev/null)"
        nvvm_mma_base_cg2="$(awk '/^define .*@compile_tcgen05_mma_base_cg2\(/,/^}/' "${nvvm_ll}" 2>/dev/null)"
        nvvm_mma_base="${nvvm_mma_base_cg1}"$'\n'"${nvvm_mma_base_cg2}"
        nvvm_mma_ws="$(awk '/^define .*@compile_tcgen05_mma_ws\(/,/^}/' "${nvvm_ll}" 2>/dev/null)"
        nvvm_mma_base_count="$(grep -oE "${mma_base_re}" <<<"${nvvm_mma_base}" | wc -l)"
        nvvm_mma_base_unique="$(grep -oE "${mma_base_re}" <<<"${nvvm_mma_base}" | sort -u | wc -l)"
        nvvm_mma_ws_count="$(grep -oE "${mma_ws_re}" <<<"${nvvm_mma_ws}" | wc -l)"
        nvvm_mma_ws_unique="$(grep -oE "${mma_ws_re}" <<<"${nvvm_mma_ws}" | sort -u | wc -l)"
        nvvm_mma_inline_count="$(grep -cE 'call void asm sideeffect ".*tcgen05\.mma' <<<"${nvvm_mma_base}"$'\n'"${nvvm_mma_ws}")"
        nvvm_mma_memory_count="$(grep -E 'call void asm sideeffect ".*tcgen05\.mma' <<<"${nvvm_mma_base}"$'\n'"${nvvm_mma_ws}" | grep -cF '~{memory}')"
        local -a nvvm_mma_attrs=()
        local nvvm_mma_attr_line
        while IFS= read -r nvvm_mma_attr_line; do
            nvvm_mma_attrs+=("${nvvm_mma_attr_line}")
        done < <(
            sed -nE '/call void asm sideeffect ".*tcgen05\.mma/s/.* (#[0-9]+)$/\1/p' \
                <<<"${nvvm_mma_base}"$'\n'"${nvvm_mma_ws}"
        )
        local nvvm_mma_convergent=1 mma_attr mma_attr_definition
        if [[ ${#nvvm_mma_attrs[@]} -ne 25 ]]; then
            nvvm_mma_convergent=0
        else
            for mma_attr in "${nvvm_mma_attrs[@]}"; do
                mma_attr_definition="$(grep -E "^attributes ${mma_attr} = \\{[^}]*\\}$" "${nvvm_ll}")"
                if [[ $(wc -l <<<"${mma_attr_definition}") -ne 1 ]] \
                    || ! grep -qw convergent <<<"${mma_attr_definition}"; then
                    nvvm_mma_convergent=0
                    break
                fi
            done
        fi
        if [[ -z "${nvvm_mma_base_cg1}" || -z "${nvvm_mma_base_cg2}" || -z "${nvvm_mma_ws}" ]] \
            || grep -qE '\.cta_group::2\.' <<<"${nvvm_mma_base_cg1}" \
            || grep -qE '\.cta_group::1\.' <<<"${nvvm_mma_base_cg2}" \
            || [[ ${nvvm_mma_base_count} -ne 9 || ${nvvm_mma_base_unique} -ne 8 ]] \
            || [[ ${nvvm_mma_ws_count} -ne 16 || ${nvvm_mma_ws_unique} -ne 10 ]] \
            || [[ ${nvvm_mma_inline_count} -ne 25 || ${nvvm_mma_memory_count} -ne 25 ]] \
            || [[ ${nvvm_mma_convergent} -ne 1 ]] \
            || [[ $(grep -oE "${mma_base_plain_re}" <<<"${nvvm_mma_base}" | wc -l) -ne 6 ]] \
            || [[ $(grep -oE "${mma_base_sp_re}" <<<"${nvvm_mma_base}" | wc -l) -ne 3 ]] \
            || [[ $(grep -oE "${mma_ws_plain_re}" <<<"${nvvm_mma_ws}" | wc -l) -ne 12 ]] \
            || [[ $(grep -oE "${mma_ws_sp_re}" <<<"${nvvm_mma_ws}" | wc -l) -ne 4 ]] \
            || [[ $(grep -cE "${mma_base_plain_re}[[:space:]]+${nvvm_shared_a_re}" <<<"${nvvm_mma_base}") -ne 4 ]] \
            || [[ $(grep -cE "${mma_base_plain_re}[[:space:]]+${nvvm_tensor_a_re}" <<<"${nvvm_mma_base}") -ne 2 ]] \
            || [[ $(grep -cE "${mma_base_sp_re}[[:space:]]+${nvvm_shared_a_re}" <<<"${nvvm_mma_base}") -ne 1 ]] \
            || [[ $(grep -cE "${mma_base_sp_re}[[:space:]]+${nvvm_tensor_a_re}" <<<"${nvvm_mma_base}") -ne 2 ]] \
            || [[ $(grep -cE "${mma_ws_plain_re}[[:space:]]+${nvvm_shared_a_re}" <<<"${nvvm_mma_ws}") -ne 2 ]] \
            || [[ $(grep -cE "${mma_ws_plain_re}[[:space:]]+${nvvm_tensor_a_re}" <<<"${nvvm_mma_ws}") -ne 10 ]] \
            || [[ $(grep -cE "${mma_ws_sp_re}[[:space:]]+${nvvm_shared_a_re}" <<<"${nvvm_mma_ws}") -ne 2 ]] \
            || [[ $(grep -cE "${mma_ws_sp_re}[[:space:]]+${nvvm_tensor_a_re}" <<<"${nvvm_mma_ws}") -ne 2 ]] \
            || [[ $(grep -oE '\.cta_group::1\.' <<<"${nvvm_mma_base}" | wc -l) -ne 5 ]] \
            || [[ $(grep -oE '\.cta_group::2\.' <<<"${nvvm_mma_base}" | wc -l) -ne 4 ]] \
            || [[ $(grep -oE '\.cta_group::1\.' <<<"${nvvm_mma_ws}" | wc -l) -ne 16 ]] \
            || grep -qE '\.cta_group::2\.' <<<"${nvvm_mma_ws}" \
            || [[ $(grep -oE '\.kind::f16\.' <<<"${nvvm_mma_base}" | wc -l) -ne 3 ]] \
            || [[ $(grep -oE '\.kind::tf32\.' <<<"${nvvm_mma_base}" | wc -l) -ne 2 ]] \
            || [[ $(grep -oE '\.kind::f8f6f4\.' <<<"${nvvm_mma_base}" | wc -l) -ne 2 ]] \
            || [[ $(grep -oE '\.kind::i8\.' <<<"${nvvm_mma_base}" | wc -l) -ne 2 ]] \
            || [[ $(grep -oE '\.kind::f16\.' <<<"${nvvm_mma_ws}" | wc -l) -ne 3 ]] \
            || [[ $(grep -oE '\.kind::tf32\.' <<<"${nvvm_mma_ws}" | wc -l) -ne 3 ]] \
            || [[ $(grep -oE '\.kind::f8f6f4\.' <<<"${nvvm_mma_ws}" | wc -l) -ne 8 ]] \
            || [[ $(grep -oE '\.kind::i8\.' <<<"${nvvm_mma_ws}" | wc -l) -ne 2 ]] \
            || [[ $(grep -oE '\.collector::a::discard' <<<"${nvvm_mma_base}" | wc -l) -ne 3 ]] \
            || [[ $(grep -oE '\.collector::a::lastuse' <<<"${nvvm_mma_base}" | wc -l) -ne 2 ]] \
            || [[ $(grep -oE '\.collector::a::fill' <<<"${nvvm_mma_base}" | wc -l) -ne 2 ]] \
            || [[ $(grep -oE '\.collector::a::use' <<<"${nvvm_mma_base}" | wc -l) -ne 2 ]] \
            || [[ $(grep -oE '\.collector::b0::discard' <<<"${nvvm_mma_ws}" | wc -l) -ne 7 ]] \
            || [[ $(grep -oE '\.collector::b1::(lastuse|fill)' <<<"${nvvm_mma_ws}" | wc -l) -ne 3 ]] \
            || [[ $(grep -oE '\.collector::b2::(fill|use)' <<<"${nvvm_mma_ws}" | wc -l) -ne 3 ]] \
            || [[ $(grep -oE '\.collector::b3::(use|discard)' <<<"${nvvm_mma_ws}" | wc -l) -ne 3 ]] \
            || [[ $(grep -oE '\.ashift' <<<"${nvvm_mma_base}" | wc -l) -ne 2 ]] \
            || grep -qE '\.ashift' <<<"${nvvm_mma_ws}" \
            || [[ $(grep -cE "${mma_ws_re}.*${nvvm_zero_mask_re}" <<<"${nvvm_mma_ws}") -ne 4 ]] \
            || [[ $(grep -cE "${mma_ws_plain_re}[[:space:]]+${nvvm_shared_a_re}.*${nvvm_zero_mask_re}" <<<"${nvvm_mma_ws}") -ne 1 ]] \
            || [[ $(grep -cE "${mma_ws_plain_re}[[:space:]]+${nvvm_tensor_a_re}.*${nvvm_zero_mask_re}" <<<"${nvvm_mma_ws}") -ne 1 ]] \
            || [[ $(grep -cE "${mma_ws_sp_re}[[:space:]]+${nvvm_shared_a_re}.*${nvvm_zero_mask_re}" <<<"${nvvm_mma_ws}") -ne 1 ]] \
            || [[ $(grep -cE "${mma_ws_sp_re}[[:space:]]+${nvvm_tensor_a_re}.*${nvvm_zero_mask_re}" <<<"${nvvm_mma_ws}") -ne 1 ]] \
            || grep -qE "${unresolved_mma_re}" "${nvvm_ll}"; then
            printf 'tcgen05 libNVVM route expected 9 base and 16 convergent, side-effecting MMA calls across all forms; got %s/%s and %s/%s total/unique\n' \
                "${nvvm_mma_base_count}" "${nvvm_mma_base_unique}" "${nvvm_mma_ws_count}" "${nvvm_mma_ws_unique}" >>"${log}"
            CARGO_EC=1
        fi

        local nvvm_ld nvvm_ld_count nvvm_ld_unique nvvm_ld_pack16 nvvm_ld_raw
        nvvm_ld="$(awk '/^define .*@compile_tcgen05_ld\(/,/^}/' "${nvvm_ll}" 2>/dev/null)"
        nvvm_ld_count="$(grep -oE "${ld_re}" <<<"${nvvm_ld}" | wc -l)"
        nvvm_ld_unique="$(grep -oE "${ld_re}" <<<"${nvvm_ld}" | sort -u | wc -l)"
        nvvm_ld_pack16="$(grep -oE "${ld_pack16_re}" <<<"${nvvm_ld}" | wc -l)"
        nvvm_ld_raw="$(grep -oE "${ld_raw_re}" <<<"${nvvm_ld}" | wc -l)"
        if [[ -z "${nvvm_ld}" ]] \
            || [[ "$(head -n 1 "${nvvm_target}" 2>/dev/null)" != "sm_100a" ]] \
            || [[ ${nvvm_ld_count} -ne 58 || ${nvvm_ld_unique} -ne 58 ]] \
            || [[ ${nvvm_ld_pack16} -ne 29 || ${nvvm_ld_raw} -ne 29 ]] \
            || grep -q 'llvm\.nvvm\.tcgen05\.ld' "${nvvm_ll}"; then
            printf 'tcgen05 libNVVM route expected sm_100a and 58 unique inline loads (29 raw and 29 pack16); got %s/%s total/unique and %s/%s raw/pack16\n' \
                "${nvvm_ld_count}" "${nvvm_ld_unique}" "${nvvm_ld_raw}" "${nvvm_ld_pack16}" >>"${log}"
            CARGO_EC=1
        fi

        local nvvm_st nvvm_st_count nvvm_st_unique nvvm_st_unpack16 nvvm_st_raw
        nvvm_st="$(awk '/^define .*@compile_tcgen05_st\(/,/^}/' "${nvvm_ll}" 2>/dev/null)"
        nvvm_st_count="$(grep -oE "${st_re}" <<<"${nvvm_st}" | wc -l)"
        nvvm_st_unique="$(grep -oE "${st_re}" <<<"${nvvm_st}" | sort -u | wc -l)"
        nvvm_st_unpack16="$(grep -oE "${st_unpack16_re}" <<<"${nvvm_st}" | wc -l)"
        nvvm_st_raw="$(grep -oE "${st_raw_re}" <<<"${nvvm_st}" | wc -l)"
        if [[ -z "${nvvm_st}" ]] \
            || [[ ${nvvm_st_count} -ne 58 || ${nvvm_st_unique} -ne 58 ]] \
            || [[ ${nvvm_st_unpack16} -ne 29 || ${nvvm_st_raw} -ne 29 ]] \
            || grep -q 'llvm\.nvvm\.tcgen05\.st' "${nvvm_ll}"; then
            printf 'tcgen05 libNVVM route expected 58 unique inline stores (29 raw and 29 unpack16); got %s/%s total/unique and %s/%s raw/unpack16\n' \
                "${nvvm_st_count}" "${nvvm_st_unique}" "${nvvm_st_raw}" "${nvvm_st_unpack16}" >>"${log}"
            CARGO_EC=1
        fi

        local nvvm_ld_offset nvvm_ld_offset_count nvvm_ld_offset_unique nvvm_ld_offset_pack16 nvvm_ld_offset_raw
        nvvm_ld_offset="$(awk '/^define .*@compile_tcgen05_ld_offset\(/,/^}/' "${nvvm_ll}" 2>/dev/null)"
        nvvm_ld_offset_count="$(grep -oE "${ld_offset_re}" <<<"${nvvm_ld_offset}" | wc -l)"
        nvvm_ld_offset_unique="$(grep -oE "${ld_offset_re}" <<<"${nvvm_ld_offset}" | sort -u | wc -l)"
        nvvm_ld_offset_pack16="$(grep -oE "${ld_offset_pack16_re}" <<<"${nvvm_ld_offset}" | wc -l)"
        nvvm_ld_offset_raw="$(grep -oE "${ld_offset_raw_re}" <<<"${nvvm_ld_offset}" | wc -l)"
        if [[ -z "${nvvm_ld_offset}" ]] \
            || [[ ${nvvm_ld_offset_count} -ne 16 || ${nvvm_ld_offset_unique} -ne 16 ]] \
            || [[ ${nvvm_ld_offset_pack16} -ne 8 || ${nvvm_ld_offset_raw} -ne 8 ]] \
            || [[ $(grep 'asm sideeffect' <<<"${nvvm_ld_offset}" | grep -o 'i64 16' | wc -l) -ne 16 ]] \
            || grep -q 'llvm\.nvvm\.tcgen05\.ld' <<<"${nvvm_ld_offset}"; then
            printf 'tcgen05 libNVVM route expected 16 unique inline offset loads with immediate 16; got %s/%s total/unique and %s/%s raw/pack16\n' \
                "${nvvm_ld_offset_count}" "${nvvm_ld_offset_unique}" "${nvvm_ld_offset_raw}" "${nvvm_ld_offset_pack16}" >>"${log}"
            CARGO_EC=1
        fi

        local nvvm_st_offset nvvm_st_offset_count nvvm_st_offset_unique nvvm_st_offset_unpack16 nvvm_st_offset_raw
        nvvm_st_offset="$(awk '/^define .*@compile_tcgen05_st_offset\(/,/^}/' "${nvvm_ll}" 2>/dev/null)"
        nvvm_st_offset_count="$(grep -oE "${st_offset_re}" <<<"${nvvm_st_offset}" | wc -l)"
        nvvm_st_offset_unique="$(grep -oE "${st_offset_re}" <<<"${nvvm_st_offset}" | sort -u | wc -l)"
        nvvm_st_offset_unpack16="$(grep -oE "${st_offset_unpack16_re}" <<<"${nvvm_st_offset}" | wc -l)"
        nvvm_st_offset_raw="$(grep -oE "${st_offset_raw_re}" <<<"${nvvm_st_offset}" | wc -l)"
        if [[ -z "${nvvm_st_offset}" ]] \
            || [[ ${nvvm_st_offset_count} -ne 16 || ${nvvm_st_offset_unique} -ne 16 ]] \
            || [[ ${nvvm_st_offset_unpack16} -ne 8 || ${nvvm_st_offset_raw} -ne 8 ]] \
            || [[ $(grep 'asm sideeffect' <<<"${nvvm_st_offset}" | grep -o 'i64 16' | wc -l) -ne 16 ]] \
            || grep -q 'llvm\.nvvm\.tcgen05\.st' <<<"${nvvm_st_offset}"; then
            printf 'tcgen05 libNVVM route expected 16 unique inline offset stores with immediate 16; got %s/%s total/unique and %s/%s raw/unpack16\n' \
                "${nvvm_st_offset_count}" "${nvvm_st_offset_unique}" "${nvvm_st_offset_raw}" "${nvvm_st_offset_unpack16}" >>"${log}"
            CARGO_EC=1
        fi
        return
    fi

    # This example gates scalar arithmetic, Ampere MMA, and extended min/max
    # on both backends.
    if [[ ${COMPILE_ONLY} -eq 1 && "${ex}" == "generated_intrinsics" ]]; then
        local llvm_ptx="crates/rustc-codegen-cuda/examples/${ex}/${ex}.ptx"
        local -a llvm_args=("build" "${ex}" "--arch=sm_86")
        local llvm_ec
        if [[ ${VERBOSE} -eq 1 ]]; then
            cargo oxide "${llvm_args[@]}" 2>&1 | tee "${log}"
            llvm_ec=${PIPESTATUS[0]}
        else
            cargo oxide "${llvm_args[@]}" >"${log}" 2>&1
            llvm_ec=$?
        fi
        if [[ ${llvm_ec} -ne 0 ]]; then
            CARGO_EC=${llvm_ec}
            return
        fi

        local scalar_ptx_re='((mul|div)\.(rn|rz|rm|rp)(\.ftz)?\.f32|fma\.(rn|rz|rm|rp)(\.ftz)?(\.sat)?\.f32|add\.(rn|rz|rm|rp)(\.sat)?(\.ftz)?\.f32|(mul|div|fma|add)\.(rn|rz|rm|rp)\.f64)'
        local ampere_float_mma_re='mma\.sync\.aligned\.(m16n8k4\.row\.col\.f32\.tf32\.tf32\.f32|m16n8k8\.row\.col\.f16\.f16\.f16\.f16|m16n8k8\.row\.col\.f32\.bf16\.bf16\.f32|m16n8k8\.row\.col\.f32\.f16\.f16\.f32|m16n8k16\.row\.col\.f16\.f16\.f16\.f16)'
        local extended_minmax_re='(min|max)(\.ftz)?(\.NaN)?(\.xorsign\.abs)?\.(f32|f16x2|bf16x2)'
        local scalar_ptx_body scalar_ptx_count scalar_ptx_unique
        local mma_ptx_body mma_ptx_count mma_ptx_unique
        local minmax_ptx_body minmax_ptx_count minmax_ptx_unique
        scalar_ptx_body="$(awk '/^\.visible \.entry compile_scalar_explicit_rounding\(/,/^}/' "${llvm_ptx}" 2>/dev/null)"
        scalar_ptx_count="$(grep -oE "${scalar_ptx_re}" <<<"${scalar_ptx_body}" | wc -l)"
        scalar_ptx_unique="$(grep -oE "${scalar_ptx_re}" <<<"${scalar_ptx_body}" | sort -u | wc -l)"
        mma_ptx_body="$(awk '/^\.visible \.entry compile_register_mma\(/,/^}/' "${llvm_ptx}" 2>/dev/null)"
        mma_ptx_count="$(grep -oE "${ampere_float_mma_re}" <<<"${mma_ptx_body}" | wc -l)"
        mma_ptx_unique="$(grep -oE "${ampere_float_mma_re}" <<<"${mma_ptx_body}" | sort -u | wc -l)"
        minmax_ptx_body="$(awk '/^\.visible \.entry compile_extended_minmax\(/,/^}/' "${llvm_ptx}" 2>/dev/null)"
        minmax_ptx_count="$(grep -oE "${extended_minmax_re}" <<<"${minmax_ptx_body}" | wc -l)"
        minmax_ptx_unique="$(grep -oE "${extended_minmax_re}" <<<"${minmax_ptx_body}" | sort -u | wc -l)"
        if [[ ! -s "${llvm_ptx}" || -z "${scalar_ptx_body}" || -z "${mma_ptx_body}" || -z "${minmax_ptx_body}" ]] \
            || ! grep -qx '\.target sm_86' "${llvm_ptx}" \
            || [[ ${scalar_ptx_count} -ne 64 || ${scalar_ptx_unique} -ne 64 ]] \
            || [[ ${mma_ptx_count} -ne 5 || ${mma_ptx_unique} -ne 5 ]] \
            || [[ ${minmax_ptx_count} -ne 28 || ${minmax_ptx_unique} -ne 28 ]] \
            || grep -qE '(\.extern|call)[^;]*llvm[.$]nvvm[.$](mul|div|fma|add)[.$](rn|rz|rm|rp)' "${llvm_ptx}"; then
            printf 'generated_intrinsics LLVM route expected 64 scalar, 5 Ampere MMA, and 28 extended min/max forms; got %s/%s, %s/%s, and %s/%s total/unique\n' \
                "${scalar_ptx_count}" "${scalar_ptx_unique}" \
                "${mma_ptx_count}" "${mma_ptx_unique}" \
                "${minmax_ptx_count}" "${minmax_ptx_unique}" >>"${log}"
            CARGO_EC=1
            return
        fi

        local -a nvvm_args=("emit-ltoir" "${ex}" "--arch=sm_86")
        if [[ ${VERBOSE} -eq 1 ]]; then
            cargo oxide "${nvvm_args[@]}" 2>&1 | tee -a "${log}"
            CARGO_EC=${PIPESTATUS[0]}
        else
            cargo oxide "${nvvm_args[@]}" >>"${log}" 2>&1
            CARGO_EC=$?
        fi
        if [[ ${CARGO_EC} -ne 0 ]]; then
            return
        fi

        local nvvm_ll="crates/rustc-codegen-cuda/examples/${ex}/${ex}.ll"
        local scalar_ir_body mma_ir_body minmax_ir_body mma_ir_count mma_ir_unique
        scalar_ir_body="$(awk '/^define .*@compile_scalar_explicit_rounding\(/,/^}/' "${nvvm_ll}" 2>/dev/null)"
        mma_ir_body="$(awk '/^define .*@compile_register_mma\(/,/^}/' "${nvvm_ll}" 2>/dev/null)"
        minmax_ir_body="$(awk '/^define .*@compile_extended_minmax\(/,/^}/' "${nvvm_ll}" 2>/dev/null)"
        mma_ir_count="$(grep -oE "${ampere_float_mma_re}" <<<"${mma_ir_body}" | wc -l)"
        mma_ir_unique="$(grep -oE "${ampere_float_mma_re}" <<<"${mma_ir_body}" | sort -u | wc -l)"
        local f32_binary_re='call float asm "((mul|div)\.(rn|rz|rm|rp)(\.ftz)?|add\.(rn|rz|rm|rp)(\.sat)?(\.ftz)?)\.f32 \$0, \$1, \$2;", "=f,f,f"'
        local f32_fma_re='call float asm "fma\.(rn|rz|rm|rp)(\.ftz)?(\.sat)?\.f32 \$0, \$1, \$2, \$3;", "=f,f,f,f"'
        local f64_binary_re='call double asm "(mul|div|add)\.(rn|rz|rm|rp)\.f64 \$0, \$1, \$2;", "=d,d,d"'
        local f64_fma_re='call double asm "fma\.(rn|rz|rm|rp)\.f64 \$0, \$1, \$2, \$3;", "=d,d,d,d"'
        local f32_minmax_re='call float asm "(min|max)(\.ftz)?(\.NaN)?(\.xorsign\.abs)?\.f32 \$0, \$1, \$2;", "=f,f,f"'
        local packed_minmax_re='call i32 asm "(min|max)(\.ftz)?(\.NaN)?(\.xorsign\.abs)?\.(f16x2|bf16x2) \$0, \$1, \$2;", "=r,r,r"'
        local f32_binary_count f32_binary_unique f32_fma_count f32_fma_unique
        local f64_binary_count f64_binary_unique f64_fma_count f64_fma_unique
        local f32_minmax_count f32_minmax_unique packed_minmax_count packed_minmax_unique
        f32_binary_count="$(grep -oE "${f32_binary_re}" <<<"${scalar_ir_body}" | wc -l)"
        f32_binary_unique="$(grep -oE "${f32_binary_re}" <<<"${scalar_ir_body}" | sort -u | wc -l)"
        f32_fma_count="$(grep -oE "${f32_fma_re}" <<<"${scalar_ir_body}" | wc -l)"
        f32_fma_unique="$(grep -oE "${f32_fma_re}" <<<"${scalar_ir_body}" | sort -u | wc -l)"
        f64_binary_count="$(grep -oE "${f64_binary_re}" <<<"${scalar_ir_body}" | wc -l)"
        f64_binary_unique="$(grep -oE "${f64_binary_re}" <<<"${scalar_ir_body}" | sort -u | wc -l)"
        f64_fma_count="$(grep -oE "${f64_fma_re}" <<<"${scalar_ir_body}" | wc -l)"
        f64_fma_unique="$(grep -oE "${f64_fma_re}" <<<"${scalar_ir_body}" | sort -u | wc -l)"
        f32_minmax_count="$(grep -oE "${f32_minmax_re}" <<<"${minmax_ir_body}" | wc -l)"
        f32_minmax_unique="$(grep -oE "${f32_minmax_re}" <<<"${minmax_ir_body}" | sort -u | wc -l)"
        packed_minmax_count="$(grep -oE "${packed_minmax_re}" <<<"${minmax_ir_body}" | wc -l)"
        packed_minmax_unique="$(grep -oE "${packed_minmax_re}" <<<"${minmax_ir_body}" | sort -u | wc -l)"
        if [[ ! -s "${nvvm_ll}" || -z "${scalar_ir_body}" || -z "${mma_ir_body}" || -z "${minmax_ir_body}" ]] \
            || [[ ${f32_binary_count} -ne 32 || ${f32_binary_unique} -ne 32 ]] \
            || [[ ${f32_fma_count} -ne 16 || ${f32_fma_unique} -ne 16 ]] \
            || [[ ${f64_binary_count} -ne 12 || ${f64_binary_unique} -ne 12 ]] \
            || [[ ${f64_fma_count} -ne 4 || ${f64_fma_unique} -ne 4 ]] \
            || [[ ${mma_ir_count} -ne 5 || ${mma_ir_unique} -ne 5 ]] \
            || [[ ${f32_minmax_count} -ne 8 || ${f32_minmax_unique} -ne 8 ]] \
            || [[ ${packed_minmax_count} -ne 20 || ${packed_minmax_unique} -ne 20 ]] \
            || grep -qE 'llvm\.nvvm\.(mul|div|fma|add)\.(rn|rz|rm|rp)|llvm\.nvvm\.f(min|max)' "${nvvm_ll}"; then
            printf 'generated_intrinsics libNVVM route expected exact scalar forms plus 5 Ampere MMA and 28 extended min/max forms; got scalar %s/%s, %s/%s, %s/%s, %s/%s, MMA %s/%s, and min/max %s/%s + %s/%s total/unique\n' \
                "${f32_binary_count}" "${f32_binary_unique}" \
                "${f32_fma_count}" "${f32_fma_unique}" \
                "${f64_binary_count}" "${f64_binary_unique}" \
                "${f64_fma_count}" "${f64_fma_unique}" \
                "${mma_ir_count}" "${mma_ir_unique}" \
                "${f32_minmax_count}" "${f32_minmax_unique}" \
                "${packed_minmax_count}" "${packed_minmax_unique}" >>"${log}"
            CARGO_EC=1
        fi
        return
    fi

    # Designated NVVM examples use `emit-ltoir` in compile-only mode so CI
    # checks both textual export and real libNVVM compilation. Other examples
    # use `build`.
    if [[ ${COMPILE_ONLY} -eq 1 ]] && verify_nvvm_in_compile_only "${ex}"; then
        local nvvm_arch
        nvvm_arch="$(nvvm_verify_arch "${ex}")"
        local -a args=("emit-ltoir" "${ex}" "--arch=${nvvm_arch}")
        if [[ ${VERBOSE} -eq 1 ]]; then
            cargo oxide "${args[@]}" 2>&1 | tee "${log}"
            CARGO_EC=${PIPESTATUS[0]}
        else
            cargo oxide "${args[@]}" >"${log}" 2>&1
            CARGO_EC=$?
        fi
        return
    fi

    local verb="run"
    if [[ ${COMPILE_ONLY} -eq 1 ]]; then verb="build"; fi
    local -a args=("${verb}" "${ex}")
    if [[ ${COMPILE_ONLY} -eq 0 && "${ex}" == "ex2_approx_f16" \
        && "${host_cc}" =~ ^[0-9]+\.[0-9]+$ \
        && $((10#${host_cc//./})) -lt 75 ]]; then
        # Build at the instruction floor; the host check skips before module loading.
        args+=("--arch=sm_75")
    fi
    if [[ "${ex}" == "debug" ]]; then
        # Its code-shape gate checks LLVM/PTX source locations, which only
        # exist when the example is compiled with full device debug metadata.
        args+=("--device-debug")
    fi
    if [[ "${ex}" == "shared_debug" ]]; then
        # Its kernel-local shared statics become function-scoped DWARF
        # variables, whose accepted placement differs across LLVM majors.
        # Only a full-debug build makes llc verify that graph, and the
        # backend now fails the build when llc rejects it instead of
        # silently emitting PTX without debug info.
        args+=("--device-debug")
    fi
    if [[ ${COMPILE_ONLY} -eq 1 ]]; then
        case "${ex}" in
            cluster) args+=("--arch=sm_90") ;;
        esac
    fi
    if [[ "${cat}" == "iket" ]]; then
        if [[ ${COMPILE_ONLY} -eq 1 ]]; then
            # CI lane: pinned floor, artifact-only bar (verdict_compile).
            args+=("--arch=sm_90")
        elif [[ ${IKET_EXEC} -eq 1 ]]; then
            args+=("--arch=${IKET_ARCH}")
        else
            # No capable GPU: fall back to a floor-pinned compile-only build.
            args=("build" "${ex}" "--arch=${IKET_ARCH}")
        fi
    fi
    if [[ ${COMPILE_ONLY} -eq 1 && "${ex}" == "interop_cubin_identity" ]]; then
        # Cubin-kind interop artifacts require a deliberate target, and a
        # GPU-less compile-only run has no detected device to satisfy it.
        # Any concrete arch works: the finalizer (libNVVM + nvJitLink) runs
        # here too, so this lane exercises the full native-artifact path.
        args+=("--arch=sm_90")
    fi
    # kind::mxf8f6f4 admits only sm_120/sm_121 in the generated target
    # gating, so the device build must always pin sm_120a; the example
    # itself decides at runtime whether the host GPU can execute it.
    if [[ "${cat}" == "blackwell-mma" ]]; then
        args+=("--arch=sm_120a")
    fi
    if [[ "${cat}" == "ltoir" || ( "${cat}" == "auto-nvvm" && ${COMPILE_ONLY} -eq 1 ) ]]; then
        args+=("--emit-nvvm-ir" "--arch=${LTOIR_ARCH}")
    fi
    if [[ "${cat}" == "ltoir-modern" ]]; then
        args+=("--emit-nvvm-ir" "--arch=${LTOIR_MODERN_ARCH}")
    fi
    if [[ ${VERBOSE} -eq 1 ]]; then
        cargo oxide "${args[@]}" 2>&1 | tee "${log}"
        CARGO_EC=${PIPESTATUS[0]}
    else
        cargo oxide "${args[@]}" >"${log}" 2>&1
        CARGO_EC=$?
    fi
    # Any example may ship a verify-code-shape.sh; running whichever exist keeps
    # a new one from being added and then silently never executed. Presence, not
    # the executable bit, is the trigger -- a script committed without +x would
    # otherwise be skipped in exactly the silence this is meant to remove -- so
    # it runs through `bash` rather than being executed directly.
    local shape_check="crates/rustc-codegen-cuda/examples/${ex}/verify-code-shape.sh"
    if [[ ${CARGO_EC} -eq 0 && -f "${shape_check}" ]]; then
        local no_opt_shape
        for no_opt_shape in "${NO_OPT_SHAPE_EXAMPLES[@]}"; do
            if [[ "${ex}" == "${no_opt_shape}" ]]; then
                # Re-emit unoptimized artifacts for the check, reusing this
                # example's own flags so the rebuild differs from the build
                # above only in optimization. The optimized build already ran
                # and already set CARGO_EC.
                local -a no_opt_args=("${args[@]}")
                no_opt_args[0]="build"
                CUDA_OXIDE_NO_OPT=1 cargo oxide "${no_opt_args[@]}" >>"${log}" 2>&1 \
                    || CARGO_EC=$?
                break
            fi
        done
    fi
    if [[ ${CARGO_EC} -eq 0 && -f "${shape_check}" ]]; then
        if ! bash "${shape_check}" >>"${log}" 2>&1; then
            printf '%s failed its verify-code-shape.sh assertions\n' "${ex}" >>"${log}"
            CARGO_EC=1
        fi
    fi
    if [[ ${CARGO_EC} -eq 0 && ${COMPILE_ONLY} -eq 1 && "${ex}" == "helper_fn" ]]; then
        local ptx="crates/rustc-codegen-cuda/examples/${ex}/${ex}.ptx"
        local llvm="crates/rustc-codegen-cuda/examples/${ex}/${ex}.ll"
        local nested_defs entry_count
        # The nested helpers are private, so the middle-end internalizes,
        # inlines, and deletes them from the optimized PTX. Their canonical
        # mangling and non-kernel classification (the #358 regression) are
        # pinned in the pre-opt LLVM IR; the PTX pins the single kernel entry.
        nested_defs="$(grep -cE '^define .*_RI.*nested_identity' "${llvm}" 2>/dev/null)"
        entry_count="$(grep -cE '^\.visible \.entry ' "${ptx}" 2>/dev/null)"
        if [[ ! -s "${ptx}" || ! -s "${llvm}" || ${nested_defs} -ne 2 || ${entry_count} -ne 1 ]] \
            || ! grep -qF '.visible .entry vecadd_with_helper(' "${ptx}" \
            || grep -qE '(nested_identity.*_TID_|_TID_.*nested_identity)' "${llvm}" \
            || grep -qE '(nested_identity.*_TID_|_TID_.*nested_identity)' "${ptx}"; then
            printf 'helper_fn expected one kernel entry, two canonically mangled nested_identity helpers in the pre-opt LLVM IR, and no _TID_ helper symbols\n' >>"${log}"
            CARGO_EC=1
        fi
    fi
    if [[ ${CARGO_EC} -eq 0 && "${ex}" == "disjoint_slice_len" ]]; then
        local llvm_ir="crates/rustc-codegen-cuda/examples/${ex}/${ex}.ll"
        local kernel_ir loaded_slice extracted_len
        kernel_ir="$(
            awk '
                /^define ptx_kernel void @write_len\(/ { in_function = 1 }
                in_function { print }
                in_function && /^}/ { exit }
            ' "${llvm_ir}" 2>/dev/null
        )"
        loaded_slice="$(
            sed -nE 's/^[[:space:]]*(%[^ ]+) = load \{ ptr, i64 \}, ptr .*/\1/p' \
                <<<"${kernel_ir}"
        )"
        extracted_len="$(
            sed -nE 's/^[[:space:]]*%[^ ]+ = extractvalue \{ ptr, i64 \} (%[^,]+), 1$/\1/p' \
                <<<"${kernel_ir}"
        )"
        if [[ ! -s "${llvm_ir}" || -z "${kernel_ir}" \
            || -z "${loaded_slice}" || -z "${extracted_len}" \
            || "$(wc -l <<<"${loaded_slice}")" -ne 1 \
            || "${extracted_len}" != "${loaded_slice}" ]]; then
            printf 'disjoint_slice_len must load the &DisjointSlice receiver before extracting field 1; the no-inline regression path was bypassed\n' >>"${log}"
            CARGO_EC=1
        fi
    fi
    if [[ ${CARGO_EC} -eq 0 && ${COMPILE_ONLY} -eq 1 && "${ex}" == "standalone_device_fn" ]]; then
        local ptx="crates/rustc-codegen-cuda/examples/${ex}/${ex}.ptx"
        local tf32_count
        tf32_count="$(grep -oF 'cvt.rna.tf32.f32' "${ptx}" 2>/dev/null | wc -l)"
        if [[ ! -s "${ptx}" || ${tf32_count} -ne 6 ]] \
            || grep -qE '(\.extern.*f2tf32|call[^;]*f2tf32)' "${ptx}"; then
            printf 'standalone_device_fn did not emit exactly 6 direct generated TF32 conversions\n' >>"${log}"
            CARGO_EC=1
        fi
    fi
    if [[ ${CARGO_EC} -eq 0 && ${COMPILE_ONLY} -eq 1 && "${ex}" == "cluster" ]]; then
        local ptx="crates/rustc-codegen-cuda/examples/cluster/cluster.ptx"
        local body cluster_count cluster_unique ncluster_count ncluster_unique mad_count mul_count store_count
        body="$(awk '/^\.visible \.entry compile_cluster_grid_helpers\(/,/^}/' "${ptx}" 2>/dev/null)"
        cluster_count="$(grep -oE '%clusterid\.[xyz]' <<<"${body}" | wc -l)"
        cluster_unique="$(grep -oE '%clusterid\.[xyz]' <<<"${body}" | sort -u | wc -l)"
        ncluster_count="$(grep -oE '%nclusterid\.[xyz]' <<<"${body}" | wc -l)"
        ncluster_unique="$(grep -oE '%nclusterid\.[xyz]' <<<"${body}" | sort -u | wc -l)"
        mad_count="$(grep -c 'mad\.lo\.s32' <<<"${body}")"
        mul_count="$(grep -c 'mul\.lo\.s32' <<<"${body}")"
        store_count="$(grep -c 'st\.global\.b32' <<<"${body}")"
        if [[ ! -s "${ptx}" || -z "${body}" ]] \
            || ! grep -qx '\.target sm_90' "${ptx}" \
            || [[ ${cluster_count} -ne 3 || ${cluster_unique} -ne 3 ]] \
            || [[ ${ncluster_count} -ne 3 || ${ncluster_unique} -ne 3 ]] \
            || [[ ${mad_count} -ne 2 || ${mul_count} -ne 2 || ${store_count} -ne 2 ]]; then
            printf 'cluster helper did not emit the exact 6-read, 2-mad, 2-mul formula\n' >>"${log}"
            if [[ ${VERBOSE} -eq 1 ]]; then
                printf 'cluster helper did not emit the exact 6-read, 2-mad, 2-mul formula\n'
            fi
            CARGO_EC=1
        fi
    fi
}

# ---- Shared example target dir -------------------------------------------
#
# Each example under crates/rustc-codegen-cuda/examples/ is its own standalone
# cargo workspace (the codegen backend is swapped in via RUSTFLAGS, so they
# can't live in the root [workspace]). By default every `cargo oxide run`
# materializes its own target/ and recompiles the whole shared dependency tree
# (cuda-device, cuda-host, proc-macros, bindgen, ...) from scratch — the
# dominant cost of this script. Point all example builds at one shared
# CARGO_TARGET_DIR: cargo fingerprints each unit by package + features +
# workspace_root + toolchain, so identical deps built with the same pinned
# nightly + backend RUSTFLAGS compile exactly once and are reused across every
# example. (Same trick the clippy CI job uses for these workspaces.)
#
# The codegen backend .so is built FIRST, with CARGO_TARGET_DIR explicitly
# cleared, so it lands at its fixed path (crates/rustc-codegen-cuda/target/
# debug) where cargo-oxide looks for it. A CARGO_TARGET_DIR in scope during
# that build would redirect the .so into the shared dir and break backend
# discovery. `cargo oxide setup` is a fast no-op when the backend is current.
printf "%sBuilding codegen backend (one-time; fast if current)...%s\n" "${C_DIM}" "${C_RESET}"
if ! env -u CARGO_TARGET_DIR cargo oxide setup >/dev/null 2>&1; then
    echo "error: failed to build the codegen backend; run 'cargo oxide setup' to see why" >&2
    exit 2
fi

# Pin the backend .so for the whole run. cargo-oxide's discovery step 1
# (CUDA_OXIDE_BACKEND) short-circuits before any cargo invocation; without
# it, every example build re-runs a cargo freshness check on the backend
# workspace (~1s each on CI runners, ~4 minutes across 219 examples). The
# `setup` above just built this exact .so, and pinning it also means the
# whole run tests one backend even if its sources change mid-run. An
# externally-set CUDA_OXIDE_BACKEND is honored as-is; if the expected path
# is missing (unexpected host layout), fall back to per-invocation
# discovery rather than failing.
if [[ -z "${CUDA_OXIDE_BACKEND:-}" ]]; then
    host_triple="$(rustc -vV 2>/dev/null | sed -n 's/^host: //p')"
    for backend_name in librustc_codegen_cuda.so librustc_codegen_cuda.dylib; do
        backend_so="${repo_root}/crates/rustc-codegen-cuda/target/${host_triple}/debug/${backend_name}"
        if [[ -n "${host_triple}" && -f "${backend_so}" ]]; then
            export CUDA_OXIDE_BACKEND="${backend_so}"
            printf "Backend for this run: %s\n" "${CUDA_OXIDE_BACKEND}"
            break
        fi
    done
fi

# Honor an externally-set CARGO_TARGET_DIR (e.g. CI); otherwise share one under
# the repo's target/ so it is gitignored and cleaned by `cargo clean`.
: "${CARGO_TARGET_DIR:=${repo_root}/target/oxide-examples}"
export CARGO_TARGET_DIR
printf "Examples share CARGO_TARGET_DIR=%s\n\n" "${CARGO_TARGET_DIR}"

# ---- Main loop -----------------------------------------------------------

log_dir="${SMOKETEST_LOG_DIR:-.smoketest-logs}"
mkdir -p "${log_dir}"

pass=0
failures=()
started=${SECONDS}
i=0
full_debug_direct_ran=0
full_debug_direct_pass=0
full_debug_configured_ran=0
full_debug_configured_pass=0

for ex in "${selected[@]}"; do
    i=$((i + 1))
    cat="$(classify "${ex}")"
    log="${log_dir}/${ex}.log"
    : > "${log}"

    if [[ ${VERBOSE} -eq 1 ]]; then
        printf "%s[%2d/%2d] %s (%s)%s\n" "${C_BOLD}" "${i}" "${total}" "${ex}" "${cat}" "${C_RESET}"
    else
        printf "[%2d/%2d] %-32s ... " "${i}" "${total}" "${ex}"
    fi

    t0=${SECONDS}
    full_debug_route=""
    if [[ ${FULL_DEBUG} -eq 1 ]]; then
        if full_debug_uses_configured_route "${ex}"; then
            full_debug_route="configured cubin/modern libNVVM (${LTOIR_MODERN_ARCH})"
            full_debug_configured_ran=$((full_debug_configured_ran + 1))
        else
            full_debug_route="direct LLVM"
            full_debug_direct_ran=$((full_debug_direct_ran + 1))
        fi
    fi
    run_cargo "${ex}" "${log}" "${cat}"
    ec=${CARGO_EC}
    dt=$((SECONDS - t0))

    if [[ ! -f "${log}" ]]; then
        verdict="FAIL (log missing: ${log})"
        status=1
    elif [[ ( ${COMPILE_ONLY} -eq 1 || ${FULL_DEBUG} -eq 1 || "${cat}" == "blackwell-compile" || "${cat}" == "sm100-compile" ) && "${cat}" != "error" ]]; then
        # Artifact-only modes collapse the GPU-gated categories: with nothing
        # executed, "PTX (or the explicitly configured artifact) compiled" is
        # the bar except for error examples, which must still fail with their
        # intended diagnostic.
        verdict="$(verdict_compile "${ex}" "${log}" "${ec}")" && status=0 || status=$?
    else
        case "${cat}" in
            error)       verdict="$(verdict_error       "${log}" "${ec}" "${ex}")" && status=0 || status=$? ;;
            tcgen05)     verdict="$(verdict_tcgen05     "${ex}" "${log}" "${ec}")" && status=0 || status=$? ;;
            wgmma)       verdict="$(verdict_wgmma       "${ex}" "${log}" "${ec}")" && status=0 || status=$? ;;
            blackwell-mma) verdict="$(verdict_blackwell_mma "${ex}" "${log}" "${ec}")" && status=0 || status=$? ;;
            ltoir)       verdict="$(verdict_ltoir       "${ex}" "${log}" "${ec}")" && status=0 || status=$? ;;
            ltoir-modern) verdict="$(verdict_ltoir_modern "${ex}" "${log}" "${ec}")" && status=0 || status=$? ;;
            auto-nvvm)   verdict="$(verdict_ltoir       "${ex}" "${log}" "${ec}")" && status=0 || status=$? ;;
            iket)        verdict="$(verdict_iket        "${ex}" "${log}" "${ec}")" && status=0 || status=$? ;;
            standard)    verdict="$(verdict_standard    "${log}" "${ec}" "${ex}")" && status=0 || status=$? ;;
            *)           verdict="FAIL (unknown category: ${cat})"; status=1 ;;
        esac
    fi

    if [[ ${FULL_DEBUG} -eq 1 ]]; then
        verdict="${verdict}; route: ${full_debug_route}"
        if [[ ${status} -eq 0 ]]; then
            if full_debug_uses_configured_route "${ex}"; then
                full_debug_configured_pass=$((full_debug_configured_pass + 1))
            else
                full_debug_direct_pass=$((full_debug_direct_pass + 1))
            fi
        fi
    fi

    if [[ ${status} -eq 0 ]]; then
        color="${C_PASS}"
    else
        color="${C_FAIL}"
    fi

    if [[ ${VERBOSE} -eq 1 ]]; then
        printf "  => %s%s%s %s[%ds]%s\n" "${color}" "${verdict}" "${C_RESET}" "${C_DIM}" "${dt}" "${C_RESET}"
    else
        printf "%s%s%s %s[%ds]%s\n" "${color}" "${verdict}" "${C_RESET}" "${C_DIM}" "${dt}" "${C_RESET}"
    fi

    if [[ ${status} -eq 0 ]]; then
        pass=$((pass + 1))
        if [[ ${KEEP_LOGS} -eq 0 ]]; then
            rm -f "${log}"
        fi
    else
        failures+=("${ex}|${verdict}|${log}")
        if [[ ${FAIL_FAST} -eq 1 ]]; then
            break
        fi
    fi
done

elapsed=$((SECONDS - started))
ran=${i}
fail=$((ran - pass))

# ---- Summary -------------------------------------------------------------

echo ""
printf "%s===== SMOKETEST SUMMARY =====%s\n" "${C_BOLD}" "${C_RESET}"
printf "Pass:    %s%d%s / %d\n" "${C_PASS}" "${pass}" "${C_RESET}" "${ran}"
printf "Fail:    %s%d%s / %d\n" "${C_FAIL}" "${fail}" "${C_RESET}" "${ran}"
if [[ ${ran} -lt ${total} ]]; then
    printf "Skipped: %s%d%s (fail-fast stopped early)\n" "${C_SKIP}" "$((total - ran))" "${C_RESET}"
fi
printf "Elapsed: %ds\n" "${elapsed}"
if [[ ${FULL_DEBUG} -eq 1 ]]; then
    printf "Full-debug direct LLVM: %d / %d passed\n" \
        "${full_debug_direct_pass}" "${full_debug_direct_ran}"
    printf "Full-debug configured route: %d / %d passed" \
        "${full_debug_configured_pass}" "${full_debug_configured_ran}"
    if [[ ${full_debug_configured_ran} -gt 0 ]]; then
        printf " (%s: cubin via modern libNVVM/nvJitLink at %s; legacy sm_90 debug is unsupported)" \
            "${FULL_DEBUG_CONFIGURED_ROUTE_EXAMPLES[*]}" "${LTOIR_MODERN_ARCH}"
    fi
    printf "\n"
fi

if [[ ${#failures[@]} -gt 0 ]]; then
    echo ""
    printf "%sFailures:%s\n" "${C_BOLD}" "${C_RESET}"
    for f in "${failures[@]}"; do
        IFS='|' read -r fex fverdict flog <<<"${f}"
        printf "  %s%s%s  %s\n  %s(log: %s)%s\n" "${C_FAIL}" "${fex}" "${C_RESET}" "${fverdict}" "${C_DIM}" "${flog}" "${C_RESET}"
    done
    exit 1
fi

exit 0
