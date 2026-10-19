#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# scripts/regress-gemm-sol-schedule.sh -- schedule-fuzz regression for the
# gemm_sol TILE_INFO mailbox handshake.
#
# The producer warp republishes the TILE_INFO mailbox for the next work
# assignment; the TILE_INFO_FREE mbarrier makes it wait for all five readers
# (epilogue warps 0-3 + the MMA warp) to acknowledge first. That handshake is
# only observable under adversarial schedules, so the regression is a
# `cargo oxide fuzz-schedule` campaign, not a unit test.
#
# Default mode (pass-after): run the campaign against the current tree with
# --fail-on-finding and require a clean sweep. Exit code 0 alone is not
# enough: a baseline that *declined* to run (wrong GPU) also exits 0 without
# running a single variant, so this script additionally asserts from the
# campaign's summary.json that
#   - the baseline verdict was Usable (baseline.kind == "Pass"),
#   - every seed in the range actually produced a variant run, and
#   - AsyncProxy schedule sites were classified (> 0), proving the campaign
#     can still reach the tcgen05/TMA orderings the handshake guards (#1195);
#     a classifier regression would otherwise silently shrink coverage.
#
# --fail-before mode (opt-in, NOT for CI): prove the harness would have caught
# the original bug by rebuilding a pre-fix ref in a throwaway git worktree and
# requiring the same campaign to produce at least one CONFIRMED finding whose
# log contains the gemm_sol_clc failure marker. Only the marker is asserted,
# never exact element values: the validator itself evolves, and the old build
# carries the old validator.
#
# Hardware gating follows scripts/smoketest.sh: without a working driver and
# an sm_100 (compute capability 10.0) GPU this prints SKIP and exits 0, so
# the recipe is safe to wire into any lane.
#
# See --help for flags.

set -euo pipefail

# ---- Campaign parameters ---------------------------------------------------

# Mirror the sweep that validated the fix. --focus mbarrier biases site
# selection toward the handshake's own instructions (mbarrier arrive/wait,
# and post-#1195 the cp.async.bulk.tensor / tcgen05.commit spellings whose
# qualifiers carry "mbarrier").
ARCH="sm_100a"
FOCUS="mbarrier"
CONFIRM_RUNS=3
TIMEOUT_SECS=20

# Half-open seed range for the pass-after sweep. ~140 variants x a few
# seconds each; narrow with --seeds for a smoke lane.
DEFAULT_SEEDS="0..140"

# Pinned fail-before seeds, derived on the pr-1066 merge lineage (pre-fix ref
# = merge-base with origin/main; each confirmed 3/3 on a B200). Seeds select
# sites positionally from the analyzed PTX, so they are NOT portable across
# builds: re-derive with a full `--seeds 0..140` sweep of the pre-fix ref
# whenever ptx-schedule's classification or gemm_sol's generated PTX changes,
# and treat these purely as a fast path.
DEFAULT_FAIL_BEFORE_SEEDS="40 123"

# The failure marker every correctness fn prints on a mismatch: a prefix of
# "FAILED: gemm_sol_persistent" (4a), "FAILED: gemm_sol_clc" (4b), and the
# multicast phases' markers, so it holds for every --phase.
FAIL_MARKER="FAILED: gemm_sol"

# ---- CLI -------------------------------------------------------------------

usage() {
    cat <<'EOF'
Usage: scripts/regress-gemm-sol-schedule.sh [OPTIONS]

Schedule-fuzz regression for the gemm_sol TILE_INFO mailbox handshake.
Requires an sm_100 (compute capability 10.0) GPU; prints SKIP and exits 0
anywhere else.

OPTIONS
  -p, --phase PHASE    GEMM_SOL_PHASE for the campaign (default:
                       4b-correctness). The other correctness phases
                       (4a/4c/4d-correctness) exercise the same handshake
                       in the other persistent kernels.
      --seeds RANGE    Half-open seed range for the pass-after sweep
                       (default: 0..140), e.g. --seeds 0..20 for a smoke run.
      --fail-before [REF]
                       Also rebuild pre-fix REF (default: the merge-base
                       with origin/main) in a throwaway git worktree and
                       require the campaign to CONFIRM the original bug
                       there. Slow and build-heavy: for manual evidence
                       gathering, not CI.
      --fail-before-seeds "S1 S2 ..."
                       Seeds for the fail-before campaign (default: 40 123;
                       see the pinned-seeds comment in this script).
      --output-dir DIR Campaign artifact root (default: .regress-gemm-sol/).
  -n, --dry-run        Print the commands that would run and exit without
                       executing any campaign.
      --no-color       Disable ANSI color. Also honours the NO_COLOR env var.
  -h, --help           Show this help and exit.

EXAMPLES
  scripts/regress-gemm-sol-schedule.sh                     # pass-after sweep
  scripts/regress-gemm-sol-schedule.sh --seeds 0..20       # quick smoke
  scripts/regress-gemm-sol-schedule.sh -p 4c-correctness   # multicast phase
  scripts/regress-gemm-sol-schedule.sh --fail-before       # + pre-fix repro
EOF
}

PHASE="4b-correctness"
SEEDS="${DEFAULT_SEEDS}"
FAIL_BEFORE=0
FAIL_BEFORE_REF=""
FAIL_BEFORE_SEEDS="${DEFAULT_FAIL_BEFORE_SEEDS}"
OUTPUT_DIR=""
DRY_RUN=0
FORCE_NO_COLOR=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--phase)   [[ $# -lt 2 ]] && { echo "error: $1 requires a value" >&2; exit 2; }; PHASE="$2"; shift 2;;
        --seeds)      [[ $# -lt 2 ]] && { echo "error: $1 requires a value" >&2; exit 2; }; SEEDS="$2"; shift 2;;
        --fail-before)
            FAIL_BEFORE=1
            # The ref is optional: `--fail-before HEAD~3` pins one, bare
            # `--fail-before` derives the merge-base with origin/main.
            if [[ $# -ge 2 && "$2" != -* ]]; then FAIL_BEFORE_REF="$2"; shift 2; else shift; fi;;
        --fail-before-seeds) [[ $# -lt 2 ]] && { echo "error: $1 requires a value" >&2; exit 2; }; FAIL_BEFORE_SEEDS="$2"; shift 2;;
        --output-dir) [[ $# -lt 2 ]] && { echo "error: $1 requires a value" >&2; exit 2; }; OUTPUT_DIR="$2"; shift 2;;
        -n|--dry-run) DRY_RUN=1; shift;;
        --no-color)   FORCE_NO_COLOR=1; shift;;
        -h|--help)    usage; exit 0;;
        *)            echo "error: unknown argument: $1" >&2; usage >&2; exit 2;;
    esac
done

if ! [[ "${SEEDS}" =~ ^([0-9]+)\.\.([0-9]+)$ ]] || [[ $((10#${BASH_REMATCH[1]})) -ge $((10#${BASH_REMATCH[2]})) ]]; then
    echo "error: --seeds must be a non-empty half-open range like 0..140 (got ${SEEDS})" >&2
    exit 2
fi
SEED_COUNT=$((10#${BASH_REMATCH[2]} - 10#${BASH_REMATCH[1]}))

case "${PHASE}" in
    4a-correctness|4b-correctness|4c-correctness|4d-correctness) ;;
    *)
        echo "error: --phase must be one of 4a/4b/4c/4d-correctness (got ${PHASE})" >&2
        exit 2;;
esac

# ---- Colors ----------------------------------------------------------------

if [[ -t 1 ]] && [[ -z "${NO_COLOR:-}" ]] && [[ ${FORCE_NO_COLOR} -eq 0 ]]; then
    C_PASS=$'\e[32m'; C_FAIL=$'\e[31m'; C_SKIP=$'\e[33m'; C_BOLD=$'\e[1m'; C_RESET=$'\e[0m'
else
    C_PASS=""; C_FAIL=""; C_SKIP=""; C_BOLD=""; C_RESET=""
fi

# ---- Preflight -------------------------------------------------------------

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

if [[ ! -f "${repo_root}/../Cargo.toml" ]] || [[ ! -d "crates/rustc-codegen-cuda/examples/gemm_sol" ]]; then
    echo "error: must be run from inside the cuda-oxide repo (got ${PWD})" >&2
    exit 2
fi

# ---- Hardware gate ---------------------------------------------------------
# Same probe as smoketest.sh: nvidia-smi can be present yet broken (driver
# mismatch, sandboxes, containers) and prints its failure text to STDOUT, so
# trust it only when it exits 0 AND the compute capability parses. The
# campaign needs real kernel launches, and cargo-oxide treats a declined
# baseline as exit 0 without running one variant -- gate here instead so
# "skipped" is never mistaken for "swept clean". The gate comes before the
# toolchain preflight on purpose: on a box that cannot run the campaign at
# all, SKIP wins over "cargo not found".
host_cc=""
if gpu_query="$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null)"; then
    gpu_info="$(head -1 <<<"${gpu_query}")"
    host_cc="$(awk -F', *' '{print $2}' <<<"${gpu_info}" | tr -d '[:space:]')"
else
    gpu_info='no GPU detected'
fi

if [[ "${host_cc}" != "10.0" ]]; then
    printf "%sSKIP%s: regress-gemm-sol-schedule requires an sm_100 GPU (datacenter Blackwell); found: %s\n" \
        "${C_SKIP}" "${C_RESET}" "${gpu_info}"
    exit 0
fi

for tool in cargo git python3; do
    if ! command -v "${tool}" >/dev/null 2>&1; then
        echo "error: ${tool} not found in PATH" >&2
        exit 2
    fi
done

OUTPUT_DIR="${OUTPUT_DIR:-${repo_root}/.regress-gemm-sol}"
# Resolve to an absolute path: --fail-before runs campaigns from a throwaway
# worktree whose EXIT cleanup would otherwise delete relative artifact dirs.
mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"

printf "%sgemm_sol schedule regression%s @ %s (%s)\n" \
    "${C_BOLD}" "${C_RESET}" "$(git rev-parse --short HEAD 2>/dev/null || echo '?')" "${gpu_info}"
printf "phase: %s   seeds: %s   output: %s\n\n" "${PHASE}" "${SEEDS}" "${OUTPUT_DIR}"

# ---- Campaign runner -------------------------------------------------------

# run_campaign WORKDIR PHASE SEED_RANGE OUT_DIR FAIL_ON_FINDING(0|1)
#
# GEMM_SOL_PHASE reaches the example binary through inherited process env:
# the campaign's build and every variant run inherit it from cargo-oxide.
run_campaign() {
    local workdir="$1" phase="$2" seed_range="$3" out_dir="$4" fail_on_finding="$5"
    local -a cmd=(cargo oxide fuzz-schedule gemm_sol
        --arch "${ARCH}" --seeds "${seed_range}" --focus "${FOCUS}"
        --confirm-runs "${CONFIRM_RUNS}" --timeout-secs "${TIMEOUT_SECS}"
        --output-dir "${out_dir}")
    if [[ ${fail_on_finding} -eq 1 ]]; then cmd+=(--fail-on-finding); fi
    if [[ ${DRY_RUN} -eq 1 ]]; then
        printf 'DRY-RUN: (cd %q && GEMM_SOL_PHASE=%q %s)\n' "${workdir}" "${phase}" "${cmd[*]}"
        return 0
    fi
    (cd "${workdir}" && GEMM_SOL_PHASE="${phase}" "${cmd[@]}")
}

# assert_pass_after SUMMARY_JSON SEED_COUNT
assert_pass_after() {
    REGRESS_SUMMARY="$1" REGRESS_SEED_COUNT="$2" python3 - <<'PY'
import json, os, sys

path = os.environ["REGRESS_SUMMARY"]
expected = int(os.environ["REGRESS_SEED_COUNT"])
with open(path) as handle:
    summary = json.load(handle)

errors = []
baseline = summary["baseline"]["kind"]
if baseline != "Pass":
    errors.append(
        f"baseline verdict is {baseline!r}, not Usable -- the sweep never ran "
        "(a declined/broken baseline also exits 0 without variants)"
    )
ran = len(summary["seeds"])
if ran != expected:
    errors.append(f"{ran} variant(s) recorded, expected {expected}")
async_proxy = summary["static_sites"]["sites_by_kind"].get("AsyncProxy", 0)
if async_proxy <= 0:
    errors.append(
        "no AsyncProxy schedule sites were classified: the campaign can no "
        "longer perturb tcgen05/TMA orderings (pre-#1195 blind spot)"
    )

for error in errors:
    print(f"assert: {error}", file=sys.stderr)
if errors:
    sys.exit(1)
print(
    f"assert: baseline Usable, {ran}/{expected} variants, "
    f"{async_proxy} AsyncProxy sites"
)
PY
}

# assert_fail_before SUMMARY_JSON... -- at least one confirmed finding whose
# log carries the gemm_sol_clc failure marker across the given summaries.
assert_fail_before() {
    REGRESS_MARKER="${FAIL_MARKER}" python3 - "$@" <<'PY'
import json, os, sys

marker = os.environ["REGRESS_MARKER"]
confirmed = []
for path in sys.argv[1:]:
    with open(path) as handle:
        summary = json.load(handle)
    baseline = summary["baseline"]["kind"]
    if baseline != "Pass":
        print(f"assert: {path}: pre-fix baseline is {baseline!r}, not Usable; "
              "the campaign never ran", file=sys.stderr)
        sys.exit(1)
    for seed in summary["seeds"]:
        conf = seed.get("confirmation")
        if not (conf and conf.get("confirmed")):
            continue
        log = seed["run"]["stdout"] + seed["run"]["stderr"]
        if marker in log:
            confirmed.append(seed["seed"])

if not confirmed:
    print(f"assert: no CONFIRMED finding with {marker!r} on the pre-fix build; "
          "re-derive the pinned seeds (see --fail-before-seeds)", file=sys.stderr)
    sys.exit(1)
print(f"assert: pre-fix bug CONFIRMED under seed(s) {confirmed}")
PY
}

# ---- Pass-after: the fixed tree must sweep clean ----------------------------

printf "%s== pass-after ==%s current tree, %s, seeds %s\n" "${C_BOLD}" "${C_RESET}" "${PHASE}" "${SEEDS}"
pass_after_dir="${OUTPUT_DIR}/pass-after"
if ! run_campaign "${repo_root}" "${PHASE}" "${SEEDS}" "${pass_after_dir}" 1; then
    printf "%sFAIL%s: campaign failed on the current tree (schedule-sensitive finding, broken baseline, or build error); see %s\n" \
        "${C_FAIL}" "${C_RESET}" "${pass_after_dir}/summary.json" >&2
    exit 1
fi
if [[ ${DRY_RUN} -eq 0 ]]; then
    assert_pass_after "${pass_after_dir}/summary.json" "${SEED_COUNT}"
    printf "%sPASS%s: clean sweep on the current tree\n" "${C_PASS}" "${C_RESET}"
fi

# ---- Fail-before (opt-in): the pre-fix tree must reproduce the bug ----------

if [[ ${FAIL_BEFORE} -eq 1 ]]; then
    ref="${FAIL_BEFORE_REF:-$(git merge-base HEAD origin/main)}"
    ref_short="$(git rev-parse --short "${ref}")"
    printf "\n%s== fail-before ==%s ref %s, %s, seeds: %s\n" \
        "${C_BOLD}" "${C_RESET}" "${ref_short}" "${PHASE}" "${FAIL_BEFORE_SEEDS}"

    worktree_dir="$(mktemp -d "${TMPDIR:-/tmp}/regress-gemm-sol.XXXXXX")/tree"
    if [[ ${DRY_RUN} -eq 1 ]]; then
        printf 'DRY-RUN: git worktree add --detach %q %q\n' "${worktree_dir}" "${ref_short}"
    else
        git worktree add --detach "${worktree_dir}" "${ref}"
    fi
    cleanup_worktree() {
        if [[ ${DRY_RUN} -eq 1 ]]; then
            printf 'DRY-RUN: git worktree remove --force %q\n' "${worktree_dir}"
        else
            git worktree remove --force "${worktree_dir}" 2>/dev/null || true
        fi
        rm -rf "$(dirname "${worktree_dir}")"
    }
    trap cleanup_worktree EXIT

    # One campaign per pinned seed: fuzz-schedule takes a half-open range, and
    # the pinned seeds are sparse. No --fail-on-finding -- findings are the
    # expected outcome here; the assertion below reads the summaries instead.
    summaries=()
    for seed in ${FAIL_BEFORE_SEEDS}; do
        if ! [[ "${seed}" =~ ^[0-9]+$ ]]; then
            echo "error: --fail-before-seeds entries must be numbers (got ${seed})" >&2
            exit 2
        fi
        seed_dir="${OUTPUT_DIR}/fail-before-${ref_short}/seed-${seed}"
        run_campaign "${worktree_dir}" "${PHASE}" "${seed}..$((seed + 1))" "${seed_dir}" 0
        summaries+=("${seed_dir}/summary.json")
    done

    if [[ ${DRY_RUN} -eq 0 ]]; then
        assert_fail_before "${summaries[@]}"
        printf "%sPASS%s: pre-fix build reproduces the bug; the harness distinguishes the two trees\n" \
            "${C_PASS}" "${C_RESET}"
    fi
fi

if [[ ${DRY_RUN} -eq 1 ]]; then
    printf "\nDRY-RUN: no campaign was executed\n"
fi
