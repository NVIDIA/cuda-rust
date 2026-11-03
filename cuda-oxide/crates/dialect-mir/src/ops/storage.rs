/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! MIR storage operations.
//!
//! This module defines variable lifetime marker operations for the MIR dialect.

use pliron::{
    builtin::op_interfaces::{NOpdsInterface, NResultsInterface},
    context::Context,
    op::Op,
};
use pliron_derive::pliron_op;

// ============================================================================
// MirStorageLiveOp
// ============================================================================

/// MIR storage live operation.
///
/// Marks the start of a variable's lifetime.
/// This is a side-effect operation with no operands or results.
///
/// # Note
///
/// In LLVM IR, this corresponds to `llvm.lifetime.start` intrinsic.
/// For GPU kernels, these are typically no-ops but help with stack allocation
/// optimization.
#[pliron_op(
    name = "mir.storage_live",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>],
    verifier = "succ"
)]
pub struct MirStorageLiveOp;

// ============================================================================
// MirStorageDeadOp
// ============================================================================

/// MIR storage dead operation.
///
/// Marks the end of a variable's lifetime.
/// This is a side-effect operation with no operands or results.
///
/// # Note
///
/// In LLVM IR, this corresponds to `llvm.lifetime.end` intrinsic.
/// For GPU kernels, these are typically no-ops but help with stack allocation
/// optimization.
#[pliron_op(
    name = "mir.storage_dead",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>],
    verifier = "succ"
)]
pub struct MirStorageDeadOp;

/// Register storage operations into the given context.
pub fn register(ctx: &mut Context) {
    MirStorageLiveOp::register(ctx);
    MirStorageDeadOp::register(ctx);
}
