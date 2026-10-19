/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Lower generated classic `cp.async` operations through the selected backend.

use crate::convert::intrinsics::common::{
    call_intrinsic, inline_asm_convergent, inline_asm_sideeffect,
};
use crate::{IntrinsicBackend, context};
use llvm_export::op_interfaces::CastOpInterface;
use llvm_export::{ops as llvm, types as llvm_types};
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::DialectConversionRewriter;
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;
use pliron::r#type::Typed;
use pliron::value::Value;

/// Lower one generated classic `cp.async` copy.
pub(crate) fn convert_generated_cp_async_copy(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    cache_policy: &str,
    copy_size: u32,
    has_source_size: bool,
    typed_intrinsic_name: &str,
) -> Result<()> {
    validate_copy_shape(ctx, op, cache_policy, copy_size, has_source_size)?;
    let operands: Vec<_> = op.deref(ctx).operands().collect();

    match context::lowering_options(ctx).intrinsic_backend {
        IntrinsicBackend::LlvmNvptx => lower_copy_with_llvm_intrinsic(
            ctx,
            rewriter,
            op,
            operands,
            has_source_size,
            typed_intrinsic_name,
        )?,
        IntrinsicBackend::LibNvvm => lower_copy_with_inline_ptx(
            ctx,
            rewriter,
            op,
            operands,
            cache_policy,
            copy_size,
            has_source_size,
        )?,
    }

    rewriter.erase_operation(ctx, op);
    Ok(())
}

/// Lower one generated classic `cp.async` control operation.
pub(crate) fn convert_generated_cp_async_control(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    operation: &str,
    typed_intrinsic_name: &str,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    let expected_operands = match operation {
        "commit_group" | "wait_all" => 0,
        "wait_group" => 1,
        _ => {
            return pliron::input_err_noloc!(
                "unsupported generated cp.async control operation `{operation}`"
            );
        }
    };
    if operands.len() != expected_operands || op.deref(ctx).get_num_results() != 0 {
        return pliron::input_err_noloc!(
            "cp.async.{operation} requires {expected_operands} operand(s) and no results"
        );
    }
    if operation == "wait_group" {
        require_i32(ctx, operands[0], "cp.async.wait_group")?;
    }

    match context::lowering_options(ctx).intrinsic_backend {
        IntrinsicBackend::LlvmNvptx => {
            lower_control_with_llvm_intrinsic(ctx, rewriter, op, operands, typed_intrinsic_name)?;
        }
        IntrinsicBackend::LibNvvm => {
            lower_control_with_inline_ptx(ctx, rewriter, op, operands, operation);
        }
    }

    rewriter.erase_operation(ctx, op);
    Ok(())
}

/// Lower one generated bridge from this thread's classic copies to an mbarrier.
pub(crate) fn convert_generated_cp_async_mbarrier(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    operation: &str,
    state_space: &str,
    typed_intrinsic_name: &str,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 1 || op.deref(ctx).get_num_results() != 0 {
        return pliron::input_err_noloc!(
            "cp.async mbarrier bridge requires one operand and no results"
        );
    }

    let (template, output_address_space) = match (operation, state_space) {
        ("arrive", "generic") => (
            "cp.async.mbarrier.arrive.b64 [$0];",
            llvm_types::address_space::GENERIC,
        ),
        ("arrive", "shared") => (
            "cp.async.mbarrier.arrive.shared.b64 [$0];",
            llvm_types::address_space::SHARED,
        ),
        ("arrive_no_inc", "generic") => (
            "cp.async.mbarrier.arrive.noinc.b64 [$0];",
            llvm_types::address_space::GENERIC,
        ),
        ("arrive_no_inc", "shared") => (
            "cp.async.mbarrier.arrive.noinc.shared.b64 [$0];",
            llvm_types::address_space::SHARED,
        ),
        _ => {
            return pliron::input_err_noloc!(
                "unsupported cp.async mbarrier bridge `{operation}` in `{state_space}` space"
            );
        }
    };
    let barrier = normalize_copy_pointer(
        ctx,
        rewriter,
        operands[0],
        llvm_types::address_space::SHARED,
        output_address_space,
        "mbarrier address",
    )?;
    let void_ty = llvm_types::VoidType::get(ctx);

    match context::lowering_options(ctx).intrinsic_backend {
        IntrinsicBackend::LlvmNvptx => {
            let pointer_ty = llvm_types::PointerType::get(ctx, output_address_space);
            let function_ty =
                llvm_types::FuncType::get(ctx, void_ty.into(), vec![pointer_ty.into()], false);
            call_intrinsic(
                ctx,
                rewriter,
                op,
                typed_intrinsic_name,
                function_ty,
                vec![barrier],
            )?;
        }
        IntrinsicBackend::LibNvvm => {
            inline_asm_convergent(
                ctx,
                rewriter,
                op,
                void_ty.into(),
                vec![barrier],
                template,
                "l,~{memory}",
            );
        }
    }

    rewriter.erase_operation(ctx, op);
    Ok(())
}

fn validate_copy_shape(
    ctx: &Context,
    op: Ptr<Operation>,
    cache_policy: &str,
    copy_size: u32,
    has_source_size: bool,
) -> Result<()> {
    let valid_variant = matches!((cache_policy, copy_size), ("ca", 4 | 8 | 16) | ("cg", 16));
    if !valid_variant {
        return pliron::input_err_noloc!(
            "unsupported classic cp.async variant: cache `{cache_policy}`, size {copy_size}"
        );
    }

    let expected_operands = if has_source_size { 3 } else { 2 };
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != expected_operands || op.deref(ctx).get_num_results() != 0 {
        return pliron::input_err_noloc!(
            "cp.async.{cache_policy}.{copy_size} requires {expected_operands} operand(s) and no results"
        );
    }
    if has_source_size {
        require_i32(ctx, operands[2], "cp.async source size")?;
    }
    Ok(())
}

fn require_i32(ctx: &Context, value: Value, name: &str) -> Result<()> {
    let ty = value.get_type(ctx);
    let is_i32 = ty
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .is_some_and(|integer| integer.width() == 32);
    if !is_i32 {
        return pliron::input_err_noloc!("{name} requires an i32 operand");
    }
    Ok(())
}

fn lower_copy_with_llvm_intrinsic(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    operands: Vec<Value>,
    has_source_size: bool,
    intrinsic_name: &str,
) -> Result<()> {
    let shared_pointer = normalize_copy_pointer(
        ctx,
        rewriter,
        operands[0],
        llvm_types::address_space::SHARED,
        llvm_types::address_space::SHARED,
        "destination",
    )?;
    let global_pointer = normalize_copy_pointer(
        ctx,
        rewriter,
        operands[1],
        llvm_types::address_space::GLOBAL,
        llvm_types::address_space::GLOBAL,
        "source",
    )?;

    let shared_ty = llvm_types::PointerType::get(ctx, llvm_types::address_space::SHARED);
    let global_ty = llvm_types::PointerType::get(ctx, llvm_types::address_space::GLOBAL);
    let mut argument_types = vec![shared_ty.into(), global_ty.into()];
    let mut arguments = vec![shared_pointer, global_pointer];
    if has_source_size {
        let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
        argument_types.push(i32_ty.into());
        arguments.push(operands[2]);
    }

    let void_ty = llvm_types::VoidType::get(ctx);
    let function_ty = llvm_types::FuncType::get(ctx, void_ty.into(), argument_types, false);
    call_intrinsic(ctx, rewriter, op, intrinsic_name, function_ty, arguments)?;
    Ok(())
}

fn lower_copy_with_inline_ptx(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    operands: Vec<Value>,
    cache_policy: &str,
    copy_size: u32,
    has_source_size: bool,
) -> Result<()> {
    let destination = normalize_copy_pointer(
        ctx,
        rewriter,
        operands[0],
        llvm_types::address_space::SHARED,
        llvm_types::address_space::GENERIC,
        "destination",
    )?;
    let source = normalize_copy_pointer(
        ctx,
        rewriter,
        operands[1],
        llvm_types::address_space::GLOBAL,
        llvm_types::address_space::GENERIC,
        "source",
    )?;
    let mut inputs = vec![destination, source];
    if has_source_size {
        inputs.push(operands[2]);
    }

    let source_size = if has_source_size { ", $2" } else { "" };
    let constraints = if has_source_size {
        "l,l,r,~{memory}"
    } else {
        "l,l,~{memory}"
    };
    let void_ty = llvm_types::VoidType::get(ctx);
    inline_asm_sideeffect(
        ctx,
        rewriter,
        op,
        void_ty.into(),
        inputs,
        &format!(
            "{{ \
            .reg .u64 %smem64; \
            .reg .u32 %smem32; \
            .reg .u64 %gmem64; \
            cvta.to.shared.u64 %smem64, $0; \
            cvt.u32.u64 %smem32, %smem64; \
            cvta.to.global.u64 %gmem64, $1; \
            cp.async.{cache_policy}.shared.global [%smem32], [%gmem64], {copy_size}{source_size}; \
            }}"
        ),
        constraints,
    );
    Ok(())
}

fn normalize_copy_pointer(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    pointer: Value,
    specific_address_space: u32,
    output_address_space: u32,
    role: &str,
) -> Result<Value> {
    let pointer_ty = pointer.get_type(ctx);
    let address_space = {
        let pointer_ty = pointer_ty.deref(ctx);
        let Some(pointer_ty) = pointer_ty.downcast_ref::<llvm_types::PointerType>() else {
            return pliron::input_err_noloc!("cp.async {role} requires an LLVM pointer");
        };
        pointer_ty.address_space()
    };

    if address_space != llvm_types::address_space::GENERIC
        && address_space != specific_address_space
    {
        return pliron::input_err_noloc!(
            "cp.async {role} requires address space {} or {}, got {}",
            llvm_types::address_space::GENERIC,
            specific_address_space,
            address_space
        );
    }
    if address_space == output_address_space {
        return Ok(pointer);
    }

    let output_ty = llvm_types::PointerType::get(ctx, output_address_space);
    let cast = llvm::AddrSpaceCastOp::new(ctx, pointer, output_ty.into());
    rewriter.insert_operation(ctx, cast.get_operation());
    Ok(cast.get_operation().deref(ctx).get_result(0))
}

fn lower_control_with_llvm_intrinsic(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    operands: Vec<Value>,
    intrinsic_name: &str,
) -> Result<()> {
    let argument_types = if operands.is_empty() {
        vec![]
    } else {
        vec![IntegerType::get(ctx, 32, Signedness::Signless).into()]
    };
    let void_ty = llvm_types::VoidType::get(ctx);
    let function_ty = llvm_types::FuncType::get(ctx, void_ty.into(), argument_types, false);
    call_intrinsic(ctx, rewriter, op, intrinsic_name, function_ty, operands)?;
    Ok(())
}

fn lower_control_with_inline_ptx(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    operands: Vec<Value>,
    operation: &str,
) {
    let (template, constraints) = match operation {
        "commit_group" => ("cp.async.commit_group;", "~{memory}"),
        "wait_all" => ("cp.async.wait_all;", "~{memory}"),
        "wait_group" => ("cp.async.wait_group $0;", "n,~{memory}"),
        _ => unreachable!("control operation was validated"),
    };
    let void_ty = llvm_types::VoidType::get(ctx);
    inline_asm_sideeffect(
        ctx,
        rewriter,
        op,
        void_ty.into(),
        operands,
        template,
        constraints,
    );
}
