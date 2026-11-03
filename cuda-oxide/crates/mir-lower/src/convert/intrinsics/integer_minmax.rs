// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Lowering helper for generated extended integer min/max operations.
//!
//! Every variant is a pure two-operand instruction over 32-bit registers:
//! the scalar `.relu` forms operate on one `s32`, and the packed forms carry
//! an `s16x2`/`u16x2` pair in one `b32` register.

use llvm_export::ops::{self as llvm, AsmKind, InlineAsmOpExt};
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::DialectConversionRewriter;
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;

/// Lower one generated integer min/max operation to its reviewed PTX
/// instruction.
pub(crate) fn convert_generated_integer_minmax(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    ptx_mnemonic: &str,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 2 {
        return pliron::input_err_noloc!(
            "generated integer min/max operation requires 2 operands, got {}",
            operands.len()
        );
    }
    let result_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let inline_asm = llvm::InlineAsmOp::build(
        ctx,
        result_ty.into(),
        operands,
        &format!("{ptx_mnemonic} $0, $1, $2;"),
        "=r,r,r",
        AsmKind::Pure,
    );
    let asm_op = inline_asm.get_operation();
    rewriter.insert_operation(ctx, asm_op);
    rewriter.replace_operation(ctx, op, asm_op);
    Ok(())
}
