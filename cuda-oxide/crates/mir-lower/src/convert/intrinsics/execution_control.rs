/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Lowering for counted barriers, programmatic dependent launch, and register control.

use crate::convert::intrinsics::common::{
    call_intrinsic, create_i32_const, inline_asm_convergent, inline_asm_sideeffect,
};
use crate::{IntrinsicBackend, context};
use llvm_export::types as llvm_types;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::rewriter::Rewriter;
use pliron::operation::Operation;
use pliron::result::Result;

pub(crate) fn convert_counted_barrier(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
    intrinsic_name: &str,
    ptx: &str,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 2 || op.deref(ctx).get_num_results() != 0 {
        return pliron::input_err_noloc!(
            "counted CTA barrier requires two operands and no results"
        );
    }
    let void_ty = llvm_types::VoidType::get(ctx);
    if context::lowering_options(ctx).intrinsic_backend == IntrinsicBackend::LibNvvm {
        inline_asm_convergent(
            ctx,
            rewriter,
            op,
            void_ty.into(),
            operands,
            ptx,
            "r,r,~{memory}",
        );
    } else {
        let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
        let function_ty = llvm_types::FuncType::get(
            ctx,
            void_ty.into(),
            vec![i32_ty.into(), i32_ty.into()],
            false,
        );
        call_intrinsic(ctx, rewriter, op, intrinsic_name, function_ty, operands)?;
    }
    rewriter.erase_operation(ctx, op);
    Ok(())
}

pub(crate) fn convert_grid_dependency(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
    intrinsic_name: &str,
    ptx: &str,
) -> Result<()> {
    if op.deref(ctx).get_num_operands() != 0 || op.deref(ctx).get_num_results() != 0 {
        return pliron::input_err_noloc!("grid dependency control requires no operands or results");
    }
    let void_ty = llvm_types::VoidType::get(ctx);
    if context::lowering_options(ctx).intrinsic_backend == IntrinsicBackend::LibNvvm {
        inline_asm_sideeffect(ctx, rewriter, op, void_ty.into(), vec![], ptx, "");
    } else {
        let function_ty = llvm_types::FuncType::get(ctx, void_ty.into(), vec![], false);
        call_intrinsic(ctx, rewriter, op, intrinsic_name, function_ty, vec![])?;
    }
    rewriter.erase_operation(ctx, op);
    Ok(())
}

pub(crate) fn convert_setmaxnreg(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
    register_count: u32,
    intrinsic_name: &str,
    direction: &str,
) -> Result<()> {
    if !(24..=256).contains(&register_count)
        || !register_count.is_multiple_of(8)
        || op.deref(ctx).get_num_operands() != 0
        || op.deref(ctx).get_num_results() != 0
    {
        return pliron::input_err_noloc!(
            "setmaxnreg requires no operands and an immediate count in 24..=256 divisible by 8"
        );
    }
    let void_ty = llvm_types::VoidType::get(ctx);
    if context::lowering_options(ctx).intrinsic_backend == IntrinsicBackend::LibNvvm {
        let template = format!("setmaxnreg.{direction}.sync.aligned.u32 {register_count};");
        inline_asm_convergent(ctx, rewriter, op, void_ty.into(), vec![], &template, "");
    } else {
        let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
        let function_ty =
            llvm_types::FuncType::get(ctx, void_ty.into(), vec![i32_ty.into()], false);
        let count = create_i32_const(ctx, rewriter, register_count as i32);
        call_intrinsic(ctx, rewriter, op, intrinsic_name, function_ty, vec![count])?;
    }
    rewriter.erase_operation(ctx, op);
    Ok(())
}
