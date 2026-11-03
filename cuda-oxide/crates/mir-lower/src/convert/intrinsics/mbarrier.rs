/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Mbarrier lowering for Ampere and newer GPUs.
//!
//! Basic operations need sm_80. Extended operations can require newer GPUs.
//!
//! # Operations
//!
//! | Operation          | Implementation | Description                         |
//! |--------------------|----------------|-------------------------------------|
//! | `Init`             | Backend route  | Initialize barrier with thread count|
//! | `Arrive`           | Backend route  | Signal arrival                      |
//! | `ArriveNoComplete` | Backend route  | Counted arrival without completion  |
//! | `ArriveExpectTx`   | Inline PTX     | Signal arrival with expected bytes  |
//! | `TestWait`         | Inline PTX     | Non-blocking wait check             |
//! | `TryWait`          | Inline PTX     | Blocking wait with hint             |
//! | `TryWaitParity`    | Inline PTX     | Parity-based wait                   |
//! | `Inval`            | Backend route  | Invalidate barrier                  |
//! | `FenceProxyAsync`  | Inline PTX     | Memory fence                        |

use crate::convert::intrinsics::common::*;
use crate::{IntrinsicBackend, context};
use llvm_export::types as llvm_types;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::rewriter::Rewriter;
use pliron::operation::Operation;
use pliron::result::Result;
use pliron::value::DefiningEntity;

/// mbarrier.init.shared: (ptr, count) -> void
pub(crate) fn convert_init(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let void_ty = llvm_types::VoidType::get(ctx);

    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 2 {
        return pliron::input_err_noloc!("mbarrier_init requires 2 operands");
    }
    let (bar_ptr, count) = (operands[0], operands[1]);
    let bar_ptr = cast_to_shared_addrspace(ctx, rewriter, bar_ptr);

    match context::lowering_options(ctx).intrinsic_backend {
        IntrinsicBackend::LlvmNvptx => {
            let ptr_ty = llvm_types::PointerType::get(ctx, llvm_types::address_space::SHARED);
            let func_ty = llvm_types::FuncType::get(
                ctx,
                void_ty.into(),
                vec![ptr_ty.into(), i32_ty.into()],
                false,
            );
            call_intrinsic(
                ctx,
                rewriter,
                op,
                "llvm_nvvm_mbarrier_init_shared",
                func_ty,
                vec![bar_ptr, count],
            )?;
        }
        IntrinsicBackend::LibNvvm => {
            inline_asm_convergent(
                ctx,
                rewriter,
                op,
                void_ty.into(),
                vec![bar_ptr, count],
                "mbarrier.init.shared.b64 [$0], $1;",
                "l,r,~{memory}",
            );
        }
    }
    rewriter.erase_operation(ctx, op);
    Ok(())
}

/// mbarrier.arrive.shared: (ptr) -> i64
pub(crate) fn convert_arrive(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let i64_ty = IntegerType::get(ctx, 64, Signedness::Signless);
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 1 {
        return pliron::input_err_noloc!("mbarrier_arrive requires 1 operand");
    }
    let bar_ptr = cast_to_shared_addrspace(ctx, rewriter, operands[0]);

    let producer = match context::lowering_options(ctx).intrinsic_backend {
        IntrinsicBackend::LlvmNvptx => {
            let ptr_ty = llvm_types::PointerType::get(ctx, llvm_types::address_space::SHARED);
            let func_ty = llvm_types::FuncType::get(ctx, i64_ty.into(), vec![ptr_ty.into()], false);
            call_intrinsic(
                ctx,
                rewriter,
                op,
                "llvm_nvvm_mbarrier_arrive_shared",
                func_ty,
                vec![bar_ptr],
            )?
        }
        IntrinsicBackend::LibNvvm => inline_asm_convergent(
            ctx,
            rewriter,
            op,
            i64_ty.into(),
            vec![bar_ptr],
            "mbarrier.arrive.shared.b64 $0, [$1];",
            "=l,l,~{memory}",
        ),
    };
    rewriter.replace_operation(ctx, op, producer);
    Ok(())
}

/// mbarrier.arrive.noComplete.shared: (ptr, count) -> i64
pub(crate) fn convert_arrive_no_complete(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let i64_ty = IntegerType::get(ctx, 64, Signedness::Signless);
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 2 {
        return pliron::input_err_noloc!("mbarrier_arrive_no_complete requires 2 operands");
    }
    let bar_ptr = cast_to_shared_addrspace(ctx, rewriter, operands[0]);
    let count = operands[1];

    let producer = match context::lowering_options(ctx).intrinsic_backend {
        IntrinsicBackend::LlvmNvptx => {
            let ptr_ty = llvm_types::PointerType::get(ctx, llvm_types::address_space::SHARED);
            let func_ty = llvm_types::FuncType::get(
                ctx,
                i64_ty.into(),
                vec![ptr_ty.into(), i32_ty.into()],
                false,
            );
            call_intrinsic(
                ctx,
                rewriter,
                op,
                "llvm_nvvm_mbarrier_arrive_noComplete_shared",
                func_ty,
                vec![bar_ptr, count],
            )?
        }
        IntrinsicBackend::LibNvvm => inline_asm_convergent(
            ctx,
            rewriter,
            op,
            i64_ty.into(),
            vec![bar_ptr, count],
            "mbarrier.arrive.noComplete.shared.b64 $0, [$1], $2;",
            "=l,l,r,~{memory}",
        ),
    };
    rewriter.replace_operation(ctx, op, producer);
    Ok(())
}

/// mbarrier.test_wait: (ptr, token) -> i1 (inline PTX)
pub(crate) fn convert_test_wait(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);

    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 2 {
        return pliron::input_err_noloc!("mbarrier_test_wait requires 2 operands");
    }
    let bar_ptr = cast_to_shared_addrspace(ctx, rewriter, operands[0]);
    let token = operands[1];

    let asm_template =
        "{ .reg .pred %p0; mbarrier.test_wait.shared.b64 %p0, [$1], $2; selp.b32 $0, 1, 0, %p0; }";
    let asm_op = inline_asm_convergent(
        ctx,
        rewriter,
        op,
        i32_ty.into(),
        vec![bar_ptr, token],
        asm_template,
        "=r,l,l,~{memory}",
    );
    let i32_result = asm_op.deref(ctx).get_result(0);
    let trunc_op = trunc_to_i1(ctx, rewriter, i32_result);
    // trunc_to_i1 returns a Value; we need the operation that defined it
    let trunc_def_op = match trunc_op.defining_entity() {
        DefiningEntity::Op(def_op) => def_op,
        _ => unreachable!(),
    };
    rewriter.replace_operation(ctx, op, trunc_def_op);
    Ok(())
}

/// mbarrier.inval: (ptr) -> void
pub(crate) fn convert_inval(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let void_ty = llvm_types::VoidType::get(ctx);
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 1 {
        return pliron::input_err_noloc!("mbarrier_inval requires 1 operand");
    }
    let bar_ptr = cast_to_shared_addrspace(ctx, rewriter, operands[0]);

    match context::lowering_options(ctx).intrinsic_backend {
        IntrinsicBackend::LlvmNvptx => {
            let ptr_ty = llvm_types::PointerType::get(ctx, llvm_types::address_space::SHARED);
            let func_ty =
                llvm_types::FuncType::get(ctx, void_ty.into(), vec![ptr_ty.into()], false);
            call_intrinsic(
                ctx,
                rewriter,
                op,
                "llvm_nvvm_mbarrier_inval_shared",
                func_ty,
                vec![bar_ptr],
            )?;
        }
        IntrinsicBackend::LibNvvm => {
            inline_asm_convergent(
                ctx,
                rewriter,
                op,
                void_ty.into(),
                vec![bar_ptr],
                "mbarrier.inval.shared.b64 [$0];",
                "l,~{memory}",
            );
        }
    }
    rewriter.erase_operation(ctx, op);
    Ok(())
}
