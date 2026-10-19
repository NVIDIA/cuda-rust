#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Shared helpers for test runner scripts.

GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# Host crates (cuda-bindings, cuda-core, cuda-async) are members of the
# git-root workspace, not this nested cutile-rs workspace. The root
# rust-toolchain.toml is what selects stable once this cds there.
GIT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
host_cargo() {
    (cd "$GIT_ROOT" && cargo "$@")
}

OVERALL_SUCCESS=true

print_header() {
    local title="$1"
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}${title}${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

run_step() {
    local step_name="$1"
    shift

    echo -e "${YELLOW}>>> ${step_name}${NC}"
    if "$@"; then
        echo -e "${GREEN}✓ ${step_name} completed successfully${NC}"
        echo ""
        return 0
    else
        echo -e "${RED}✗ ${step_name} failed${NC}"
        echo ""
        OVERALL_SUCCESS=false
        return 1
    fi
}

run_examples() {
    local examples_dir="$1"
    shift

    # Separate cargo flags (--features ...) from skip-list names.
    local cargo_extra=()
    local skip_examples=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --features) cargo_extra+=("$1" "$2"); shift 2 ;;
            --*) cargo_extra+=("$1"); shift ;;
            *) skip_examples+=("$1"); shift ;;
        esac
    done

    local examples_passed=0
    local examples_failed=0
    local examples_skipped=0
    local failed_examples=()

    mapfile -t examples < <(find "$examples_dir" -maxdepth 1 -name '*.rs' -printf '%f\n' | sed 's/\.rs$//' | sort)

    for example in "${examples[@]}"; do
        local should_skip=false
        for skip in "${skip_examples[@]}"; do
            if [[ "$example" == "$skip" ]]; then
                should_skip=true
                break
            fi
        done
        if [[ "$should_skip" == true ]]; then
            echo "Skipping example: $example"
            ((examples_skipped++))
            continue
        fi

        echo -e "  Running example: ${example}"
        local output skip_reason
        if output=$(cargo run --example "$example" "${cargo_extra[@]}" --quiet 2>&1); then
            # Examples print this marker only for unmet runtime prerequisites.
            # A nonzero exit is always a failure, even if it printed a marker.
            skip_reason=$(printf '%s\n' "$output" | sed -n "/^SKIP ${example}: /p")
            if [[ -n "$skip_reason" ]]; then
                echo -e "  ${YELLOW}-${NC} ${skip_reason}"
                ((examples_skipped++))
            else
                echo -e "  ${GREEN}✓${NC} ${example}"
                ((examples_passed++))
            fi
        else
            echo -e "  ${RED}✗${NC} ${example}"
            printf '%s\n' "$output"
            ((examples_failed++))
            failed_examples+=("$example")
            OVERALL_SUCCESS=false
        fi
    done

    echo ""
    if [[ $examples_failed -eq 0 ]]; then
        echo -e "${GREEN}✓ ${examples_passed} examples passed, ${examples_skipped} skipped${NC}"
    else
        echo -e "${RED}✗ ${examples_failed} examples failed, ${examples_passed} passed, ${examples_skipped} skipped${NC}"
        echo -e "${RED}Failed examples: ${failed_examples[*]}${NC}"
    fi
    echo ""
}

run_benches() {
    local benches_dir="$1"
    shift

    local cargo_extra=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --features) cargo_extra+=("$1" "$2"); shift 2 ;;
            --*) cargo_extra+=("$1"); shift ;;
            *) shift ;;
        esac
    done

    local benches_passed=0
    local benches_failed=0
    local failed_benches=()

    mapfile -t benches < <(find "$benches_dir" -maxdepth 1 -name '*.rs' -printf '%f\n' | sed 's/\.rs$//' | sort)

    for bench in "${benches[@]}"; do
        echo -e "  Running benchmark: ${bench}"
        if cargo bench --bench "$bench" "${cargo_extra[@]}" --quiet >/dev/null 2>&1; then
            echo -e "  ${GREEN}✓${NC} ${bench}"
            ((benches_passed++))
        else
            echo -e "  ${RED}✗${NC} ${bench}"
            ((benches_failed++))
            failed_benches+=("$bench")
            OVERALL_SUCCESS=false
        fi
    done

    echo ""
    if [[ $benches_failed -eq 0 ]]; then
        echo -e "${GREEN}✓ All ${benches_passed} benchmarks validated${NC}"
    else
        echo -e "${RED}✗ ${benches_failed} benchmarks failed, ${benches_passed} passed${NC}"
        echo -e "${RED}Failed benchmarks: ${failed_benches[*]}${NC}"
    fi
    echo ""
}

print_summary_and_exit() {
    local success_message="$1"
    local failure_message="$2"

    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Summary${NC}"
    echo -e "${BLUE}========================================${NC}"
    if [[ "$OVERALL_SUCCESS" == true ]]; then
        echo -e "${GREEN}✓ ${success_message}${NC}"
        exit 0
    else
        echo -e "${RED}✗ ${failure_message}${NC}"
        exit 1
    fi
}
