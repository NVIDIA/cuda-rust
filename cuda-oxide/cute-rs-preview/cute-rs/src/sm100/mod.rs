/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! SM100 GEMM building blocks for a pair of thread blocks (CTAs).
//!
//! Rust types describe shared-memory layouts and the two output partitions.
//! The compiler turns these calls into CuTe operations, then uses CUTLASS to
//! produce GPU code. Call them only inside device code.
//!
//! ```text
//! global A/B -> TMA -> shared tiles -> MMA -> tensor memory -> output
//! ```

/// A shared tile used as the left input of a matrix product.
pub enum OperandA {}
/// A shared tile used as the right input of a matrix product.
pub enum OperandB {}

mod mma;
mod pipeline;
mod shared;
mod tma;
mod tmem;

pub use mma::Sm100TiledMma;
pub use pipeline::{Sm100AccumulatorPipeline, Sm100TmaMmaPipeline, Sm100TmaStorePipeline};
pub use shared::Sm100SharedTile;
pub use tma::Sm100ClusterTmaCopy;
pub use tmem::{Sm100Tmem, Sm100TmemEpilogue};

/// Calls recognized by the compiler; never execute these on the host.
#[doc(hidden)]
#[allow(missing_docs)]
pub mod __compiler;
/// Regression tests for this module’s layout, representation and rejection contracts.
#[cfg(test)]
mod tests;
