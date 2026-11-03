/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Lower generated packed integer dot products through the selected backend.

use crate::convert::intrinsics::common::{call_intrinsic, create_i1_const};
use crate::{IntrinsicBackend, context};
use llvm_export::ops::{self as llvm, AsmKind, InlineAsmOpExt};
use llvm_export::types as llvm_types;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::DialectConversionRewriter;
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;

/// Lower one generated `dp4a` or `dp2a.lo` operation.
pub(crate) fn convert_generated_dot_product(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    typed_intrinsic_name: &str,
    inline_ptx: &str,
    insert_low_half_selector: bool,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 3 || op.deref(ctx).get_num_results() != 1 {
        return pliron::input_err_noloc!(
            "generated dot product requires exactly three operands and one result"
        );
    }

    match context::lowering_options(ctx).intrinsic_backend {
        IntrinsicBackend::LlvmNvptx => lower_with_llvm_intrinsic(
            ctx,
            rewriter,
            op,
            operands,
            typed_intrinsic_name,
            insert_low_half_selector,
        ),
        IntrinsicBackend::LibNvvm => {
            let result_ty = IntegerType::get(ctx, 32, Signedness::Signless);
            let inline_asm = llvm::InlineAsmOp::build(
                ctx,
                result_ty.into(),
                operands,
                inline_ptx,
                "=r,r,r,r",
                AsmKind::Pure,
            );
            let asm_op = inline_asm.get_operation();
            rewriter.insert_operation(ctx, asm_op);
            rewriter.replace_operation(ctx, op, asm_op);
            Ok(())
        }
    }
}

fn lower_with_llvm_intrinsic(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    operands: Vec<pliron::value::Value>,
    intrinsic_name: &str,
    insert_low_half_selector: bool,
) -> Result<()> {
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let (argument_types, arguments) = if insert_low_half_selector {
        let i1_ty = IntegerType::get(ctx, 1, Signedness::Signless);
        let low = create_i1_const(ctx, rewriter, false);
        (
            vec![i32_ty.into(), i32_ty.into(), i1_ty.into(), i32_ty.into()],
            vec![operands[0], operands[1], low, operands[2]],
        )
    } else {
        (vec![i32_ty.into(), i32_ty.into(), i32_ty.into()], operands)
    };
    let function_ty = llvm_types::FuncType::get(ctx, i32_ty.into(), argument_types, false);
    let call = call_intrinsic(ctx, rewriter, op, intrinsic_name, function_ty, arguments)?;
    rewriter.replace_operation(ctx, op, call);
    Ok(())
}
