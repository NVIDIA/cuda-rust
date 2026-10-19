/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Optimization passes over the `dialect-mir` IR.
//!
//! These run in the middle of cuda-oxide's pipeline: after `mem2reg` (the pass
//! that promotes memory slots to plain SSA values) and before the IR is lowered
//! to the LLVM dialect on its way to PTX. The first pass here is loop unrolling,
//! switched on by the `#[unroll]` / `#[unroll(N)]` annotation (recorded as a
//! `mir.unroll_hint` operation inside the annotated loop). More loop passes can
//! live here too.

pub mod analyses;
mod canonicalize;
pub mod forward_compiler_result_bundles;
pub mod scalarize_borrowed_aggregate_reads;
pub mod unroll;
