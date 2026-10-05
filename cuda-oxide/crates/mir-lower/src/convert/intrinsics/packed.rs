// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared lowering helpers for generated packed arithmetic and conversions.

use crate::convert::intrinsics::common::call_intrinsic;
use crate::{IntrinsicBackend, context};
use llvm_export::ops::{self as llvm, AsmKind, InlineAsmOpExt};
use llvm_export::types as llvm_types;
use pliron::builtin::types::{FP32Type, IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::DialectConversionRewriter;
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;

/// Lower one generated packed ALU operation to its reviewed PTX instruction.
pub(crate) fn convert_generated_packed_alu(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    ptx_mnemonic: &str,
    width: u32,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    let register_constraint = match width {
        32 => "r",
        64 => "l",
        width => {
            return pliron::input_err_noloc!(
                "generated packed ALU operation requires a 32- or 64-bit carrier, got {width}"
            );
        }
    };
    let constraints = match operands.len() {
        1..=3 => std::iter::once(format!("={register_constraint}"))
            .chain(std::iter::repeat_n(
                register_constraint.to_owned(),
                operands.len(),
            ))
            .collect::<Vec<_>>()
            .join(","),
        count => {
            return pliron::input_err_noloc!(
                "generated packed ALU operation requires 1 to 3 operands, got {count}"
            );
        }
    };
    let operand_list = (0..=operands.len())
        .map(|index| format!("${index}"))
        .collect::<Vec<_>>()
        .join(", ");
    let result_ty = IntegerType::get(ctx, width, Signedness::Signless);
    let inline_asm = llvm::InlineAsmOp::build(
        ctx,
        result_ty.into(),
        operands,
        &format!("{ptx_mnemonic} {operand_list};"),
        &constraints,
        AsmKind::Pure,
    );
    let asm_op = inline_asm.get_operation();
    rewriter.insert_operation(ctx, asm_op);
    rewriter.replace_operation(ctx, op, asm_op);
    Ok(())
}

/// Convert one already-packed register to another packed format.
///
/// Covers the `f16x2` and packed-FP8 conversions, which take a single source
/// register rather than the scalar `f32` pair. PTX orders `cvt` operands as
/// `d, a`, so the lone operand needs no reordering.
pub(crate) fn convert_generated_packed_unary(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    ptx_mnemonic: &str,
    result_width: u32,
    source_width: u32,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 1 || op.deref(ctx).get_num_results() != 1 {
        return pliron::input_err_noloc!(
            "generated packed unary conversion requires one operand and one result"
        );
    }
    // `h` names a 16-bit register and `r` a 32-bit one.
    let constraint = match (result_width, source_width) {
        (16, 32) => "=h,r",
        (32, 16) => "=r,h",
        (16, 16) => "=h,h",
        (32, 32) => "=r,r",
        (result, source) => {
            return pliron::input_err_noloc!(
                "generated packed unary conversion requires 16- or 32-bit operands, got {result} from {source}"
            );
        }
    };
    let result_ty = IntegerType::get(ctx, result_width, Signedness::Signless);
    let inline_asm = llvm::InlineAsmOp::build(
        ctx,
        result_ty.into(),
        operands,
        &format!("{ptx_mnemonic} $0, $1;"),
        constraint,
        AsmKind::Pure,
    );
    let asm_op = inline_asm.get_operation();
    rewriter.insert_operation(ctx, asm_op);
    rewriter.replace_operation(ctx, op, asm_op);
    Ok(())
}

/// Pack two `f32` values, keeping the first argument in the low lane.
pub(crate) fn convert_generated_packed_f32x2(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    typed_intrinsic_name: Option<&str>,
    ptx_mnemonic: &str,
    result_width: u32,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 2 || op.deref(ctx).get_num_results() != 1 {
        return pliron::input_err_noloc!(
            "generated packed f32x2 conversion requires two operands and one result"
        );
    }
    let constraint = match result_width {
        16 => "=h,f,f",
        32 => "=r,f,f",
        width => {
            return pliron::input_err_noloc!(
                "generated packed f32x2 conversion requires a 16- or 32-bit result, got {width}"
            );
        }
    };
    let result_ty = IntegerType::get(ctx, result_width, Signedness::Signless);
    match (
        context::lowering_options(ctx).intrinsic_backend,
        typed_intrinsic_name,
    ) {
        (IntrinsicBackend::LlvmNvptx, Some(intrinsic_name)) => {
            let f32_ty = FP32Type::get(ctx);
            let function_ty = llvm_types::FuncType::get(
                ctx,
                result_ty.into(),
                vec![f32_ty.into(), f32_ty.into()],
                false,
            );
            let call = call_intrinsic(
                ctx,
                rewriter,
                op,
                intrinsic_name,
                function_ty,
                vec![operands[1], operands[0]],
            )?;
            rewriter.replace_operation(ctx, op, call);
            Ok(())
        }
        _ => {
            let inline_asm = llvm::InlineAsmOp::build(
                ctx,
                result_ty.into(),
                operands,
                &format!("{ptx_mnemonic} $0, $2, $1;"),
                constraint,
                AsmKind::Pure,
            );
            let asm_op = inline_asm.get_operation();
            rewriter.insert_operation(ctx, asm_op);
            rewriter.replace_operation(ctx, op, asm_op);
            Ok(())
        }
    }
}
