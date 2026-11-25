/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! 5th-generation Tensor Core (`tcgen05`) UMMA and TMEM copy atoms (`cutlass.cute.nvgpu.tcgen05`).

pub use crate::nvgpu::common::{OperandA, OperandB};

pub mod copy;
pub mod mma;
pub mod shared;

pub use copy::{Sm100TmemEpilogue, TmemFragment, tmem_row_address};
pub use mma::{
    CollectorDiscard, CollectorFill, CollectorLastUse, CollectorOp, CollectorOpMarker,
    CollectorUse, MmaSs, MmaTs, SM107BlockScaledMmaMXF8F6F4Op, Sm100BlockScaledTiledMma,
    Sm100TiledMma,
};
pub use shared::{
    MmaHalfElement, SharedLayout, SharedTile, SharedTileArray, Sm100ScaleTile, Sm100SharedTile,
    SmemLayout,
};

/// Calls recognized by the compiler; never execute these on the host.
#[doc(hidden)]
#[allow(missing_docs)]
pub mod __compiler;

/// Regression tests for this module's layout, representation and rejection contracts.
#[cfg(test)]
mod tests;
