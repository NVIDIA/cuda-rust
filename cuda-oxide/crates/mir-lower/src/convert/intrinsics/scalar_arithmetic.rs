/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Lowering helper for generated scalar floating-point arithmetic.

use super::common::call_intrinsic;
use crate::{IntrinsicBackend, context};
use llvm_export::{
    ops::{self as llvm, AsmKind, InlineAsmOpExt},
    types as llvm_types,
};
use pliron::{
    builtin::types::{FP32Type, FP64Type},
    context::{Context, Ptr},
    irbuild::{
        dialect_conversion::DialectConversionRewriter, inserter::Inserter, rewriter::Rewriter,
    },
    op::Op,
    operation::Operation,
    result::Result,
};

pub(crate) fn convert_generated_scalar_arithmetic(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    intrinsic_name: &str,
    ptx_mnemonic: &str,
    is_f64: bool,
    llvm_inline_ptx: bool,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if !matches!(operands.len(), 2 | 3) || op.deref(ctx).get_num_results() != 1 {
        return pliron::input_err_noloc!(
            "generated scalar arithmetic requires two or three operands and one result"
        );
    }

    let result_ty = if is_f64 {
        FP64Type::get(ctx).into()
    } else {
        FP32Type::get(ctx).into()
    };
    let backend = context::lowering_options(ctx).intrinsic_backend;
    let lowered = match backend {
        IntrinsicBackend::LlvmNvptx if !llvm_inline_ptx => {
            let function_ty =
                llvm_types::FuncType::get(ctx, result_ty, vec![result_ty; operands.len()], false);
            call_intrinsic(ctx, rewriter, op, intrinsic_name, function_ty, operands)?
        }
        IntrinsicBackend::LlvmNvptx | IntrinsicBackend::LibNvvm => {
            let constraint = match (is_f64, operands.len()) {
                (false, 2) => "=f,f,f",
                (false, 3) => "=f,f,f,f",
                (true, 2) => "=d,d,d",
                (true, 3) => "=d,d,d,d",
                _ => unreachable!("validated scalar-arithmetic arity"),
            };
            let operand_list = (0..=operands.len())
                .map(|index| format!("${index}"))
                .collect::<Vec<_>>()
                .join(", ");
            let inline_asm = llvm::InlineAsmOp::build(
                ctx,
                result_ty,
                operands,
                &format!("{ptx_mnemonic} {operand_list};"),
                constraint,
                AsmKind::Pure,
            );
            let inline_op = inline_asm.get_operation();
            rewriter.insert_operation(ctx, inline_op);
            inline_op
        }
    };
    rewriter.replace_operation(ctx, op, lowered);
    Ok(())
}
