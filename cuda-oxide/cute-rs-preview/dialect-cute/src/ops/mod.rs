/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! CuTe operation families grouped by their semantic responsibility.
//!
//! Use the family modules for tensor, copy, GEMV, MMA, pipeline, scheduler,
//! and epilogue operations. The direct copy, divisibility, matrix-load, and
//! TMA operation names remain available here for existing callers.

pub mod copy;
mod divisibility;
pub mod epilogue;
pub mod gemm_tma;
pub mod gemv;
pub mod mma;
pub mod pipeline;
pub mod scheduler;
pub mod sm100;
pub mod sm100_attention;
pub mod tensor;

pub use copy::{CuteCopyG2SOp, CuteCopyOp, CuteTmaLoad2dOp, CuteTmaStore2dOp};
pub use divisibility::CuteAssumeDivOp;
pub use mma::CuteLdmatrixOp;

/// Regression fixtures covering accepted contracts and rejected malformed IR.
#[cfg(test)]
mod tests;
