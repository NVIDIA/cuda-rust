/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compatibility operation for the former grid-sync intrinsic path.
//!
//! `cuda_device::grid::sync()` now implements grid synchronization in Rust.
//! It does not create this operation, and no active importer or lowering path
//! consumes it. [`GridSyncOp`] remains registered to preserve the public
//! dialect API.

use pliron::{
    builtin::op_interfaces::{NOpdsInterface, NResultsInterface},
    context::Context,
    context::Ptr,
    op::Op,
    operation::Operation,
};
use pliron_derive::pliron_op;

// =============================================================================
// Grid Sync
// =============================================================================

/// Public compatibility type for the former grid-sync dialect operation.
///
/// New intrinsic imports must not create this op. Grid synchronization uses
/// the Rust implementation in `cuda_device::grid::sync()`.
#[pliron_op(
    name = "nvvm.grid_sync",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>],
)]
pub struct GridSyncOp;

impl GridSyncOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        GridSyncOp { op }
    }
}

/// Register all grid-scoped operations.
pub fn register(ctx: &mut Context) {
    GridSyncOp::register(ctx);
}
