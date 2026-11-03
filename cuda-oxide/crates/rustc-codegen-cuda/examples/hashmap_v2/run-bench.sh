#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Run the `hashmap_v2` perf bench against CPU `hashbrown`.
#
# `cargo oxide run hashmap_v2` invokes the default binary (the 12-test
# correctness suite) so the smoketest harness keeps working unchanged.
# The bench is a second binary in the same crate and is opt-in through
# this script (or directly via `cargo oxide run hashmap_v2 --bin bench`).
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${script_dir}/../../../.." && pwd)"
exec cargo oxide run hashmap_v2 --bin bench "$@"
