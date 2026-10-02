#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run the atomic_f16 f32/f16 atomic-add microbenchmark.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${script_dir}/../../../.." && pwd)"
exec cargo oxide run atomic_f16 --bin bench "$@"
