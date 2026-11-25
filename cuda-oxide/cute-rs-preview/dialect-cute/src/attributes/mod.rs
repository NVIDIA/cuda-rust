/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Typed attributes for the cute dialect.
//!
//! Layouts are stored structurally (the `cute-layout` types), never as
//! strings; only the *textual syntax* is the CuTe notation in quotes, e.g.
//! `"(8,32):(32,1)"`, so module dumps read like CuTe IR.

/// Typed copy widths, divisibility promises, and warp matrix operand roles.
mod copy;
/// Shared output placement and asynchronous store synchronization contracts.
mod epilogue;
/// Packed-value and scale-view contracts for the block-scaled GEMV flow.
mod gemv;
/// Structural layouts and swizzles in the CuTe attribute syntax.
mod layout;
/// Shared tensor placement, register carriers, and tiled matrix-instruction contracts.
mod mma;
/// Barrier-ring configuration and producer/consumer cursor contracts.
mod pipeline;
/// Persistent output-tile grids and their ordinary cursor provenance.
mod scheduler;
/// Tensor storage, access, format, and selection metadata.
mod tensor;

pub use copy::{CuteCopyAtomAttr, CuteDivisibilityAttr, CuteMatrixRoleAttr};
pub use epilogue::{
    CuteCountedCtaBarrierAttr, CuteEpilogueHalfAttr, CuteEpiloguePlanAttr,
    CuteEpilogueSyncPhaseAttr, CuteTmaStorePipelineAttr,
};
pub use gemv::CuteScaledLayoutAttr;
pub use layout::{CuteComposedLayoutAttr, CuteLayoutAttr};
pub use mma::{
    CuteMmaAccumulatorAttr, CuteMmaAtomAttr, CuteMmaCarrierKindAttr, CuteTiledMmaPlanAttr,
};
pub use pipeline::{CutePipelineRoleAttr, CutePipelineStateAttr};
pub use scheduler::CuteTileGridAttr;
pub use tensor::{
    CuteAlignmentAttr, CuteTensorAccessAttr, CuteTensorAddressSpaceAttr, CuteTensorFormatAttr,
    CuteTensorLayoutAttr, CuteTensorRoleAttr, SM1XX_BLOCK_SCALE_ATOM_BYTES,
    SM1XX_BLOCK_SCALE_GROUPS_PER_ATOM, SM1XX_BLOCK_SCALE_ROWS_PER_ATOM,
};

/// Regression fixtures covering accepted contracts and rejected malformed IR.
#[cfg(test)]
mod tests;
