/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Lowering helper for generated unary scalar floating-point math.

use super::common::call_intrinsic;
use crate::{IntrinsicBackend, context};
use llvm_export::{
    ops::{self as llvm, AsmKind, InlineAsmOpExt},
    types as llvm_types,
};
use pliron::{
    builtin::types::{FP32Type, FP64Type, IntegerType, Signedness},
    context::{Context, Ptr},
    irbuild::{
        dialect_conversion::DialectConversionRewriter, inserter::Inserter, rewriter::Rewriter,
    },
    op::Op,
    operation::Operation,
    result::Result,
    r#type::Typed,
};

#[allow(clippy::too_many_arguments)]
pub(crate) fn convert_generated_scalar_math(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    intrinsic_name: &str,
    ptx_mnemonic: &str,
    is_f16: bool,
    is_f64: bool,
    llvm_inline_ptx: bool,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 1 || op.deref(ctx).get_num_results() != 1 {
        return pliron::input_err_noloc!(
            "generated scalar math requires one operand and one result"
        );
    }

    let result_ty = if is_f16 {
        IntegerType::get(ctx, 16, Signedness::Unsigned).into()
    } else if is_f64 {
        FP64Type::get(ctx).into()
    } else {
        FP32Type::get(ctx).into()
    };
    let type_matches = |ty: pliron::r#type::TypeHandle| {
        if is_f16 {
            ty.deref(ctx)
                .downcast_ref::<IntegerType>()
                .is_some_and(|integer| integer.width() == 16)
        } else if is_f64 {
            ty.deref(ctx).downcast_ref::<FP64Type>().is_some()
        } else {
            ty.deref(ctx).downcast_ref::<FP32Type>().is_some()
        }
    };
    if !type_matches(operands[0].get_type(ctx))
        || !type_matches(op.deref(ctx).get_result(0).get_type(ctx))
    {
        return pliron::input_err_noloc!(
            "generated scalar math operand and result types must match the selected format"
        );
    }
    let backend = context::lowering_options(ctx).intrinsic_backend;
    let lowered = match backend {
        IntrinsicBackend::LlvmNvptx if !llvm_inline_ptx => {
            // PTX-native scalar math ops (tanh) have no typed intrinsic and
            // are always generated with llvm_inline_ptx set, so an empty
            // name can only mean a corrupted generated table.
            if intrinsic_name.is_empty() {
                return pliron::input_err_noloc!(
                    "generated scalar math op has no typed LLVM intrinsic"
                );
            }
            let function_ty = llvm_types::FuncType::get(ctx, result_ty, vec![result_ty], false);
            call_intrinsic(ctx, rewriter, op, intrinsic_name, function_ty, operands)?
        }
        IntrinsicBackend::LlvmNvptx | IntrinsicBackend::LibNvvm => {
            let constraint = if is_f16 {
                "=h,h"
            } else if is_f64 {
                "=d,d"
            } else {
                "=f,f"
            };
            let inline_asm = llvm::InlineAsmOp::build(
                ctx,
                result_ty,
                operands,
                &format!("{ptx_mnemonic} $0, $1;"),
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
