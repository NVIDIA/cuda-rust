/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Tests build kinded fixture types directly; production minting lives in
// mir-importer's facts.rs (see the workspace clippy.toml disallowed-methods).
#![allow(clippy::disallowed_methods)]

mod allowlist;
mod atomics;
mod carriers;
mod cluster_and_memory;
mod inline_ptx;
mod matrix;
mod thread_and_warp;
