/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Block-scaled MXFP4 matrix multiply for SM120 GPUs.
//!
//! This module covers the compute part of a 128 x 128 output tile:
//!
//! ```text
//! packed FP4 A + packed FP4 B + 8-bit scales
//!                       |
//!                       v
//!                shared memory
//!                       |
//!                       v
//!          fragments held by each GPU lane
//!                       |
//!                       v
//!       m16n8k64 tensor-core instructions
//!                       |
//!                       v
//!             2 x 8 FP32 result cells
//! ```
//!
//! Words used below:
//!
//! - **M** and **N** are the output row and column dimensions. **K** is the
//!   shared dimension whose products are summed.
//! - A **warp** is 32 GPU threads running one instruction together.
//! - A **lane** is one thread inside that warp.
//! - A **fragment** is the small part of a matrix held by one lane.
//! - **MMA** means matrix multiply-accumulate: `result = A * B + result`.
//! - An **MMA cell** is one 16 x 8 result made by a tensor-core instruction.
//! - A **scale pair** contains two scales: one for each K=32 half of K=64.
//! - A **stage** is one K=128 input block held in shared memory.
//! - **TMA** is the GPU copy engine used to move whole tiles.
//! - **MXFP4** stores 4-bit E2M1 numbers with an 8-bit UE8M0 scale for each
//!   group of 32 numbers.
//!
//! These Rust types add checks and names. CuTe-aware compilers retain the
//! shared-memory MMA story and choose its implementation for their backend.

mod epilogue;
mod fragments;
mod tiled;

pub use fragments::{
    Mxf4AccumulatorTile2x8, Mxf4BTileK64, Mxf4ScalePairs, Mxf4ScalePairs128, Mxf4ScaleStage,
    Mxf4ScaleTile128,
};
pub use tiled::{Mxf4Nvf4TiledMma, Mxf4TiledMma, Nvfp4TiledMma};

/// Stable compiler boundaries for the SM120 shared-memory MMA path.
#[doc(hidden)]
pub mod __compiler;
/// Regression tests for this module’s layout, representation and rejection contracts.
#[cfg(test)]
mod tests;
