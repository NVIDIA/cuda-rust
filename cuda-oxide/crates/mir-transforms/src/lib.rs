/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Required library analysis and optional transforms over `dialect-mir`.
//!
//! The Block library proof consumes prepared SSA after `mem2reg`, or original
//! slots in Full debug, before provider execution and native ABI binding.
//! Optional passes and annotated loop unrolling run afterward, before LLVM
//! lowering.

pub mod analyses;
mod canonicalize;
pub mod forward_compiler_result_bundles;
pub mod scalarize_borrowed_aggregate_reads;
pub mod unroll;

/// Required prepared-MIR library proof and native ABI lowering.
pub mod library;
