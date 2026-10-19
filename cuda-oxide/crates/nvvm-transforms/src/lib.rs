/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Converts lowered LLVM operations to the forms accepted by libNVVM.
//!
//! The transform runs after MIR-to-LLVM lowering and before text export.
//! Pre-Blackwell targets receive LLVM 7-compatible operations. Blackwell and
//! newer targets keep modern operations, apart from NVVM-wide compatibility
//! rewrites.

#![warn(missing_docs)]

mod helpers;
mod legalize;

use llvm_export::export::NvvmIrDialect;
use pliron::{
    context::{Context, Ptr},
    operation::Operation,
    result::Result,
};

/// Legalize a lowered LLVM module for the selected NVVM input dialect.
///
/// Legacy LLVM 7 modules receive the complete compatibility pass. Modern
/// modules retain modern operations but still rewrite integer widths that
/// libNVVM does not accept in bit-manipulation intrinsics.
///
/// `capability` is the numeric compute capability of the build target
/// (`52`, `86`, `90`, ...). The legacy pass uses it to reject rewrites whose
/// PTX instructions have a hardware floor, so the failure is a clear
/// diagnostic here instead of a downstream libNVVM or ptxas error.
pub fn legalize_for_nvvm(
    ctx: &mut Context,
    module: Ptr<Operation>,
    dialect: NvvmIrDialect,
    capability: u32,
) -> Result<()> {
    match dialect {
        NvvmIrDialect::LegacyLlvm7 => legalize::legalize_for_legacy_nvvm(ctx, module, capability),
        NvvmIrDialect::Modern => legalize::legalize_nvvm_bit_intrinsics(ctx, module),
    }
}
