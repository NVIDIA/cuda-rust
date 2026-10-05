/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Lower generated Cluster Launch Control operations through typed NVVM calls.

use crate::convert::intrinsics::common::{
    call_intrinsic, cast_to_shared_addrspace, create_i64_const,
};
use llvm_export::attributes::IntegerOverflowFlagsAttr;
use llvm_export::op_interfaces::{
    ATTR_KEY_INTEGER_OVERFLOW_FLAGS, BinArithOp, CastOpWithNNegInterface,
};
use llvm_export::ops as llvm;
use llvm_export::types as llvm_types;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;
use pliron::value::Value;

/// Lower a CLC cancellation request with two shared-memory pointers.
pub(crate) fn convert_generated_clc_try_cancel(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
    intrinsic_name: &str,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 2 || op.deref(ctx).get_num_results() != 0 {
        return pliron::input_err_noloc!(
            "generated CLC try_cancel requires two operands and no result"
        );
    }

    let pointer_ty = llvm_types::PointerType::get(ctx, llvm_types::address_space::SHARED);
    let void_ty = llvm_types::VoidType::get(ctx);
    let function_ty = llvm_types::FuncType::get(
        ctx,
        void_ty.into(),
        vec![pointer_ty.into(), pointer_ty.into()],
        false,
    );
    let arguments = operands
        .into_iter()
        .map(|pointer| cast_to_shared_addrspace(ctx, rewriter, pointer))
        .collect();
    call_intrinsic(ctx, rewriter, op, intrinsic_name, function_ty, arguments)?;
    rewriter.erase_operation(ctx, op);
    Ok(())
}

/// Lower one CLC response query after packing its two u64 halves into i128.
pub(crate) fn convert_generated_clc_query(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
    intrinsic_name: &str,
    bool_result: bool,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 2 || op.deref(ctx).get_num_results() != 1 {
        return pliron::input_err_noloc!(
            "generated CLC query requires two operands and one result"
        );
    }

    let i1_ty = IntegerType::get(ctx, 1, Signedness::Signless);
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let i128_ty = IntegerType::get(ctx, 128, Signedness::Signless);
    let response = pack_u64_pair(ctx, rewriter, operands[0], operands[1], i128_ty);
    let result_ty = if bool_result {
        i1_ty.into()
    } else {
        i32_ty.into()
    };
    let function_ty = llvm_types::FuncType::get(ctx, result_ty, vec![i128_ty.into()], false);
    let call = call_intrinsic(
        ctx,
        rewriter,
        op,
        intrinsic_name,
        function_ty,
        vec![response],
    )?;

    if bool_result {
        let result = call.deref(ctx).get_result(0);
        let extend = llvm::ZExtOp::new_with_nneg(ctx, result, i32_ty.into(), false);
        rewriter.insert_operation(ctx, extend.get_operation());
        rewriter.replace_operation(ctx, op, extend.get_operation());
    } else {
        rewriter.replace_operation(ctx, op, call);
    }
    Ok(())
}

fn pack_u64_pair(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    lo: Value,
    hi: Value,
    i128_ty: pliron::r#type::TypedHandle<IntegerType>,
) -> Value {
    let lo_wide = llvm::ZExtOp::new_with_nneg(ctx, lo, i128_ty.into(), false);
    rewriter.insert_operation(ctx, lo_wide.get_operation());
    let lo_wide = lo_wide.get_operation().deref(ctx).get_result(0);

    let hi_wide = llvm::ZExtOp::new_with_nneg(ctx, hi, i128_ty.into(), false);
    rewriter.insert_operation(ctx, hi_wide.get_operation());
    let hi_wide = hi_wide.get_operation().deref(ctx).get_result(0);

    let shift = create_i64_const(ctx, rewriter, 64);
    let shift_wide = llvm::ZExtOp::new_with_nneg(ctx, shift, i128_ty.into(), false);
    rewriter.insert_operation(ctx, shift_wide.get_operation());
    let shift_wide = shift_wide.get_operation().deref(ctx).get_result(0);

    let hi_shifted = llvm::ShlOp::new(ctx, hi_wide, shift_wide);
    hi_shifted.get_operation().deref_mut(ctx).attributes.set(
        ATTR_KEY_INTEGER_OVERFLOW_FLAGS.clone(),
        IntegerOverflowFlagsAttr::default(),
    );
    rewriter.insert_operation(ctx, hi_shifted.get_operation());
    let hi_shifted = hi_shifted.get_operation().deref(ctx).get_result(0);

    let response = llvm::OrOp::new(ctx, lo_wide, hi_shifted);
    rewriter.insert_operation(ctx, response.get_operation());
    response.get_operation().deref(ctx).get_result(0)
}
