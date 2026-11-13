#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Enforce the branch naming convention in CONTRIBUTING.md.
set -euo pipefail
export LC_ALL=C

if branch=$(git symbolic-ref --quiet --short HEAD); then
    :
else
    status=$?
    if [[ $status -eq 1 ]]; then
        # Detached HEAD has no branch name to validate.
        exit 0
    fi
    exit "$status"
fi

pattern='^(feat|fix|docs|refactor|perf|test|build|ci|chore)/[^[:upper:]]+$'
if [[ ! $branch =~ $pattern ]]; then
    printf 'error: branch name "%s" does not follow CONTRIBUTING.md\n' "$branch" >&2
    printf 'Use <prefix>/<lowercase-description>, for example fix/kernel-launch-sync.\n' >&2
    printf 'Allowed prefixes: feat, fix, docs, refactor, perf, test, build, ci, chore.\n' >&2
    exit 1
fi
