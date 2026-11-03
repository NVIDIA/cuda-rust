/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Entry point for the submission-lifetime tests, kept out of the aggregate
//! `gpu` binary on purpose. Each test gates a stream with a blocking host
//! function; any context-wide synchronize elsewhere in the same process (a
//! first-use module load, a cache eviction's module unload, pool growth) then
//! waits on that gate while the gate owner waits on the context, and only the
//! Gate's safety bound breaks the cycle. `scripts/run_gpu_tests.sh` runs this
//! binary with `--test-threads=1` so no test's warm-up overlaps another's gate.

mod common;

#[path = "gpu/submission_lifetimes.rs"]
mod submission_lifetimes;
