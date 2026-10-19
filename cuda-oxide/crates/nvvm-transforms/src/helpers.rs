/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use llvm_export::{ops as llvm, types::FuncType};
use pliron::{
    basic_block::BasicBlock,
    builtin::op_interfaces::SymbolOpInterface,
    context::{Context, Ptr},
    linked_list::ContainsLinkedList,
    op::Op,
    r#type::TypedHandle,
};

pub(super) fn ensure_intrinsic_declared(
    ctx: &mut Context,
    llvm_block: Ptr<BasicBlock>,
    intrinsic_name: &str,
    func_ty: TypedHandle<FuncType>,
) -> Result<(), String> {
    let func_op = llvm_block
        .deref(ctx)
        .get_parent_op(ctx)
        .ok_or_else(|| "block has no parent operation (expected function)".to_string())?;
    let module_op = func_op
        .deref(ctx)
        .get_parent_op(ctx)
        .ok_or_else(|| "function has no parent operation (expected module)".to_string())?;
    let module_block = module_op
        .deref(ctx)
        .get_region(0)
        .deref(ctx)
        .iter(ctx)
        .next()
        .ok_or_else(|| "module region is empty".to_string())?;
    let symbol = intrinsic_name
        .try_into()
        .map_err(|error| format!("invalid intrinsic name {intrinsic_name:?}: {error:?}"))?;

    let already_declared = module_block.deref(ctx).iter(ctx).any(|existing_op| {
        pliron::operation::Operation::get_op::<llvm::FuncOp>(existing_op, ctx)
            .is_some_and(|func| func.get_symbol_name(ctx) == symbol)
    });
    if !already_declared {
        llvm::FuncOp::new(ctx, symbol, func_ty)
            .get_operation()
            .insert_before(ctx, func_op);
    }
    Ok(())
}

#[cfg(test)]
pub(super) fn create_i1_constant(
    ctx: &mut Context,
    llvm_block: Ptr<BasicBlock>,
    value: bool,
) -> Result<pliron::value::Value, String> {
    use std::num::NonZeroUsize;

    use pliron::{
        builtin::{
            attributes::IntegerAttr,
            types::{IntegerType, Signedness},
        },
        utils::apint::APInt,
    };

    let ty = IntegerType::get(ctx, 1, Signedness::Signless);
    let width = NonZeroUsize::new(1).expect("one is non-zero");
    let attr = IntegerAttr::new(ty, APInt::from_i64(i64::from(value), width));
    let constant = llvm::ConstantOp::new(ctx, Box::new(attr));
    constant.get_operation().insert_at_back(llvm_block, ctx);
    Ok(constant.get_operation().deref(ctx).get_result(0))
}
