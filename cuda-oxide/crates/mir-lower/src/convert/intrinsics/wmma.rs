/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared lowering helpers for generated matrix intrinsics.

use crate::convert::intrinsics::common::*;
use llvm_export::ops as llvm;
use llvm_export::types as llvm_types;
use pliron::builtin::types::{FP32Type, FP64Type, IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::DialectConversionRewriter;
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;

/// Result carrier used by generated MMA recipes.
#[derive(Clone, Copy)]
pub(crate) enum GeneratedMmaResultType {
    F32,
    F64,
    I32,
}

/// Lower one generated register-MMA variant.
#[allow(clippy::too_many_arguments)]
pub(crate) fn convert_generated_register_mma(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    result_type: GeneratedMmaResultType,
    result_count: usize,
    expected_operands: usize,
    template: &str,
    constraints: &str,
) -> Result<()> {
    convert_generated_mma(
        ctx,
        rewriter,
        op,
        result_type,
        result_count,
        expected_operands,
        template,
        constraints,
        "generated register MMA",
    )
}

/// Lower one generated sparse-MMA variant.
#[allow(clippy::too_many_arguments)]
pub(crate) fn convert_generated_sparse_mma(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    result_type: GeneratedMmaResultType,
    result_count: usize,
    expected_operands: usize,
    template: &str,
    constraints: &str,
) -> Result<()> {
    convert_generated_mma(
        ctx,
        rewriter,
        op,
        result_type,
        result_count,
        expected_operands,
        template,
        constraints,
        "generated sparse MMA",
    )
}

#[allow(clippy::too_many_arguments)]
fn convert_generated_mma(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    result_type: GeneratedMmaResultType,
    result_count: usize,
    expected_operands: usize,
    template: &str,
    constraints: &str,
    family: &str,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != expected_operands {
        return pliron::input_err_noloc!(
            "{family} requires {expected_operands} operands, got {}",
            operands.len()
        );
    }
    let scalar_type = match result_type {
        GeneratedMmaResultType::F32 => FP32Type::get(ctx).into(),
        GeneratedMmaResultType::F64 => FP64Type::get(ctx).into(),
        GeneratedMmaResultType::I32 => IntegerType::get(ctx, 32, Signedness::Signless).into(),
    };
    let result_type = llvm_types::StructType::get_unnamed(
        ctx,
        (
            vec![scalar_type; result_count],
            llvm_types::StructLayout::Unpacked,
        ),
    );
    let inline_asm = inline_asm_convergent(
        ctx,
        rewriter,
        op,
        result_type.into(),
        operands,
        template,
        constraints,
    );
    let aggregate = inline_asm.deref(ctx).get_result(0);
    let mut results = Vec::with_capacity(result_count);
    for index in 0..result_count {
        let extract = llvm::ExtractValueOp::new(ctx, aggregate, vec![index as u32])
            .map_err(|error| pliron::input_error_noloc!("{}", error))?;
        rewriter.insert_operation(ctx, extract.get_operation());
        results.push(extract.get_operation().deref(ctx).get_result(0));
    }
    rewriter.replace_operation_with_values(ctx, op, results);
    Ok(())
}
