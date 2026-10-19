#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run the paper-facing §5.1 B200 throughput benchmark bundle.
#
# JIT timing and exploratory sweeps are intentionally separated under
# ../diagnostics and ../tuning.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$HERE/run_b200_persistent_gemm_elemwise.sh"
