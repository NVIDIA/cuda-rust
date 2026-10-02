/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! User-authored inline PTX lowering.

use crate::convert::types::convert_type;
use dialect_nvvm::ops::InlinePtxOp;
use llvm_export::ops as llvm;
use llvm_export::types as llvm_types;
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::location::Located;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;
use pliron::r#type::{TypeHandle, Typed};

pub(crate) fn convert_inline_ptx(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let loc = op.deref(ctx).loc();
    let inline_ptx = InlinePtxOp::new(op);
    let template = inline_ptx
        .get_attr_ptx_template(ctx)
        .map(|attr| String::from((*attr).clone()))
        .ok_or_else(|| pliron::input_error!(loc.clone(), "nvvm.inline_ptx missing ptx_template"))?;
    let constraints = inline_ptx
        .get_attr_ptx_constraints(ctx)
        .map(|attr| String::from((*attr).clone()))
        .ok_or_else(|| {
            pliron::input_error!(loc.clone(), "nvvm.inline_ptx missing ptx_constraints")
        })?;
    let sideeffect = inline_ptx
        .get_attr_ptx_sideeffect(ctx)
        .map(|attr| bool::from((*attr).clone()))
        .ok_or_else(|| {
            pliron::input_error!(loc.clone(), "nvvm.inline_ptx missing ptx_sideeffect")
        })?;
    let convergent = inline_ptx
        .get_attr_ptx_convergent(ctx)
        .map(|attr| bool::from((*attr).clone()))
        .ok_or_else(|| {
            pliron::input_error!(loc.clone(), "nvvm.inline_ptx missing ptx_convergent")
        })?;

    let num_results = op.deref(ctx).get_num_results();
    let operands: Vec<_> = op.deref(ctx).operands().collect();

    match num_results {
        0 => {
            // Void: no results.
            let result_ty = llvm_types::VoidType::get(ctx).into();
            let inline_asm = llvm::InlineAsmOp::new(
                ctx,
                result_ty,
                operands,
                &template,
                &constraints,
                sideeffect,
            );
            let asm_op = inline_asm.get_operation();
            crate::convert::preserve_location(ctx, op, asm_op);
            llvm::set_inline_asm_convergent(ctx, asm_op, convergent);
            rewriter.insert_operation(ctx, asm_op);
            rewriter.erase_operation(ctx, op);
        }
        1 => {
            // Single result: backward-compatible path.
            let mir_ty = {
                let op_ref = op.deref(ctx);
                op_ref.get_result(0).get_type(ctx)
            };
            let result_ty = convert_type(ctx, mir_ty)
                .map_err(|err| pliron::input_error!(loc.clone(), "{err}"))?;
            let inline_asm = llvm::InlineAsmOp::new(
                ctx,
                result_ty,
                operands,
                &template,
                &constraints,
                sideeffect,
            );
            let asm_op = inline_asm.get_operation();
            // No explicit location copy: `replace_operation` propagates the
            // replaced op's location onto a location-less replacement.
            llvm::set_inline_asm_convergent(ctx, asm_op, convergent);
            rewriter.insert_operation(ctx, asm_op);
            rewriter.replace_operation(ctx, op, asm_op);
        }
        n => {
            // Multi-result: build an LLVM struct return type, emit
            // inline asm, then extractvalue each element.
            let mut llvm_field_types: Vec<TypeHandle> = Vec::with_capacity(n);
            for i in 0..n {
                let mir_ty = {
                    let op_ref = op.deref(ctx);
                    op_ref.get_result(i).get_type(ctx)
                };
                let llvm_ty = convert_type(ctx, mir_ty)
                    .map_err(|err| pliron::input_error!(loc.clone(), "{err}"))?;
                llvm_field_types.push(llvm_ty);
            }
            let struct_ty: TypeHandle = llvm_types::StructType::get_unnamed(
                ctx,
                (llvm_field_types, llvm_types::StructLayout::Unpacked),
            )
            .into();

            let inline_asm = llvm::InlineAsmOp::new(
                ctx,
                struct_ty,
                operands,
                &template,
                &constraints,
                sideeffect,
            );
            let asm_op = inline_asm.get_operation();
            crate::convert::preserve_location(ctx, op, asm_op);
            llvm::set_inline_asm_convergent(ctx, asm_op, convergent);
            rewriter.insert_operation(ctx, asm_op);

            let aggregate = asm_op.deref(ctx).get_result(0);

            let mut extracted_values = Vec::with_capacity(n);
            for i in 0..n {
                let extract = llvm::ExtractValueOp::new(ctx, aggregate, vec![i as u32])
                    .map_err(|error| pliron::input_error!(loc.clone(), "{}", error))?;
                crate::convert::preserve_location(ctx, op, extract.get_operation());
                rewriter.insert_operation(ctx, extract.get_operation());
                extracted_values.push(extract.get_operation().deref(ctx).get_result(0));
            }

            rewriter.replace_operation_with_values(ctx, op, extracted_values);
        }
    }

    Ok(())
}
