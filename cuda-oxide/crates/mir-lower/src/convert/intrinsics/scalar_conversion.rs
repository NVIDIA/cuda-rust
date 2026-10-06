/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Lowering helper for generated scalar conversions.

use super::common::call_intrinsic;
use crate::{IntrinsicBackend, context};
use llvm_export::{
    ops::{self as llvm, AsmKind, InlineAsmOpExt},
    types as llvm_types,
};
use pliron::{
    builtin::types::{FP32Type, IntegerType, Signedness},
    context::{Context, Ptr},
    irbuild::{
        dialect_conversion::DialectConversionRewriter, inserter::Inserter, rewriter::Rewriter,
    },
    op::Op,
    operation::Operation,
    result::Result,
};

pub(crate) fn convert_generated_scalar_conversion(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    intrinsic_name: &str,
    ptx_mnemonic: &str,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 1 || op.deref(ctx).get_num_results() != 1 {
        return pliron::input_err_noloc!(
            "generated scalar conversion requires one operand and one result"
        );
    }

    let result_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let lowered = match context::lowering_options(ctx).intrinsic_backend {
        IntrinsicBackend::LlvmNvptx => {
            let function_ty = llvm_types::FuncType::get(
                ctx,
                result_ty.into(),
                vec![FP32Type::get(ctx).into()],
                false,
            );
            call_intrinsic(ctx, rewriter, op, intrinsic_name, function_ty, operands)?
        }
        IntrinsicBackend::LibNvvm => {
            let inline_asm = llvm::InlineAsmOp::build(
                ctx,
                result_ty.into(),
                operands,
                &format!("{ptx_mnemonic} $0, $1;"),
                "=r,f",
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
