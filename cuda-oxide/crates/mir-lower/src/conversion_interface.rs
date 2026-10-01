/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! The `MirToLlvmConversion` op interface.
//!
//! Defined here (in `mir-lower`) rather than in `dialect-mir` so that the
//! `#[op_interface_impl]` blocks for MIR ops can live in this crate without
//! violating Rust's orphan rules: the trait is local, so we can implement
//! it for foreign types (`MirAddOp`, `MirSubOp`, etc.).

use pliron::{
    context::Context,
    derive::op_interface,
    irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo},
    op::Op,
    result::Result,
};

/// Op interface for `dialect-mir` → LLVM dialect lowering.
///
/// Every `dialect-mir` and `dialect-nvvm` op that the lowering pass can
/// handle implements this interface. The pass dispatches via
/// `op_cast::<dyn MirToLlvmConversion>` instead of a manual if-chain over
/// `OpId`.
///
/// Called by `DialectConversion::rewrite` via `op_cast`-based dispatch.
/// Each concrete op provides its own implementation (see
/// `convert/interface_impls.rs`). The implementation should use `rewriter`
/// to replace the current op with one or more LLVM dialect ops.
#[op_interface]
pub trait MirToLlvmConversion {
    /// Lower this operation to the LLVM dialect.
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()>;

    /// Verification hook (no-op — the underlying op verifiers are sufficient).
    fn verify(_op: &dyn Op, _ctx: &Context) -> Result<()>
    where
        Self: Sized,
    {
        Ok(())
    }
}
