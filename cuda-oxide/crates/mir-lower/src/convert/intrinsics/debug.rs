/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Debug and profiling intrinsic conversion.
//!
//! | Operation      | Lowering                                | PTX Output              |
//! |----------------|-----------------------------------------|-------------------------|
//! | `Clock`        | `llvm_nvvm_read_ptx_sreg_clock`         | `mov %r, %clock`        |
//! | `Clock64`      | `llvm_nvvm_read_ptx_sreg_clock64`       | `mov %rd, %clock64`     |
//! | `Globaltimer`  | `llvm_nvvm_read_ptx_sreg_globaltimer`   | `mov %rd, %globaltimer` |
//! | `AssertFail`   | `call @__assertfail`                    | assertion system call   |
//! | `Vprintf`      | `call @vprintf`                         | `call vprintf`          |

use crate::helpers;
use llvm_export::ops as llvm;
use llvm_export::types as llvm_types;
use pliron::builtin::op_interfaces::CallOpCallable;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::linked_list::ContainsLinkedList;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;

pub(crate) fn convert_assertfail(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 5 {
        return pliron::input_err_noloc!(
            "__assertfail requires 5 operands, got {}",
            operands.len()
        );
    }

    let void_ty = llvm_types::VoidType::get(ctx);
    let i8_ptr_ty = llvm_types::PointerType::get(ctx, 0);
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let i64_ty = IntegerType::get(ctx, 64, Signedness::Signless);

    let func_ty = llvm_types::FuncType::get(
        ctx,
        void_ty.into(),
        vec![
            i8_ptr_ty.into(),
            i8_ptr_ty.into(),
            i32_ty.into(),
            i8_ptr_ty.into(),
            i64_ty.into(),
        ],
        false,
    );

    let parent_block = op
        .deref(ctx)
        .get_parent_block()
        .ok_or_else(|| pliron::input_error_noloc!("nvvm.assertfail has no parent block"))?;

    // Everything after assertfail in the same block is unreachable.
    // Collect first, then erase in reverse order so uses disappear before defs.
    let trailing_ops: Vec<_> = parent_block
        .deref(ctx)
        .iter(ctx)
        .skip_while(|candidate| *candidate != op)
        .skip(1)
        .collect();

    let declaration =
        helpers::ensure_intrinsic_declared(ctx, parent_block, "__assertfail", func_ty)
            .map_err(|error| pliron::input_error_noloc!("{error}"))?;
    let declaration_op = Operation::get_op::<llvm::FuncOp>(declaration, ctx)
        .ok_or_else(|| pliron::input_error_noloc!("__assertfail declaration is not llvm.func"))?;
    let mut declaration_attrs = declaration_op
        .get_attr_llvm_func_attrs(ctx)
        .map(|attrs| (*attrs).clone())
        .unwrap_or_default();
    declaration_attrs.set("noreturn", llvm::LlvmAttrValue::Unit);
    declaration_op.set_attr_llvm_func_attrs(ctx, declaration_attrs);

    let sym_name: pliron::identifier::Identifier = "__assertfail".try_into().unwrap();
    let callee = CallOpCallable::Direct(sym_name);
    let call_op = llvm::CallOp::new(ctx, callee, func_ty, operands);
    let mut call_attrs = call_op
        .get_attr_llvm_call_attrs(ctx)
        .map(|attrs| (*attrs).clone())
        .unwrap_or_default();
    call_attrs.set("noreturn", llvm::LlvmAttrValue::Unit);
    call_op.set_attr_llvm_call_attrs(ctx, call_attrs);

    for dead_op in trailing_ops.into_iter().rev() {
        rewriter.erase_operation(ctx, dead_op);
    }

    rewriter.insert_operation(ctx, call_op.get_operation());

    let unreachable = llvm::UnreachableOp::new(ctx);
    rewriter.insert_operation(ctx, unreachable.get_operation());

    rewriter.erase_operation(ctx, op);

    Ok(())
}

pub(crate) fn convert_vprintf(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 2 {
        return pliron::input_err_noloc!("vprintf requires 2 operands, got {}", operands.len());
    }

    let format_ptr = operands[0];
    let args_ptr = operands[1];

    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let i8_ptr_ty = llvm_types::PointerType::get(ctx, 0);

    let func_ty = llvm_types::FuncType::get(
        ctx,
        i32_ty.into(),
        vec![i8_ptr_ty.into(), i8_ptr_ty.into()],
        false,
    );

    let parent_block = op.deref(ctx).get_parent_block().unwrap();
    helpers::ensure_intrinsic_declared(ctx, parent_block, "vprintf", func_ty)
        .map_err(|e| pliron::input_error_noloc!("{}", e))?;

    let sym_name: pliron::identifier::Identifier = "vprintf".try_into().unwrap();
    let callee = CallOpCallable::Direct(sym_name);
    let call_op = llvm::CallOp::new(ctx, callee, func_ty, vec![format_ptr, args_ptr]);
    rewriter.insert_operation(ctx, call_op.get_operation());
    rewriter.replace_operation(ctx, op, call_op.get_operation());

    Ok(())
}
