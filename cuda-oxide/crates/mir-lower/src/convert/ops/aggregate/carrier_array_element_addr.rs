/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Array-element address lowering for verified packed-AS3 carrier locals.

use super::addressing;
use crate::convert::types::{llvm_type_size_align, mir_element_stride};
use crate::packed_shared_local_storage::{carrier_gep_source_type, carrier_projection_alignment};
use dialect_mir::types::MirPtrType;
use llvm_export::ops as llvm;
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;
use pliron::r#type::Typed;

/// Lower an array element address using the exact physical element type stamped
/// by the pre-lowering carrier proof.
///
/// Ordinary array addressing remains on the established path. Carrier array
/// projections never infer storage from the converted pointer or its defining
/// operation; the typed GEP source is a private fact on this exact MIR op.
pub(crate) fn convert_array_element_addr(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    operands_info: &OperandsInfo,
) -> Result<()> {
    let Some(carrier_element_ty) = carrier_gep_source_type(ctx, op) else {
        return addressing::convert_array_element_addr(ctx, rewriter, op, operands_info);
    };

    let loc = op.deref(ctx).loc().clone();
    let arr_ptr = op.deref(ctx).get_operand(0);
    let index = op.deref(ctx).get_operand(1);
    let result_ty = op.deref(ctx).get_result(0).get_type(ctx);
    let semantic_element_ty = result_ty
        .deref(ctx)
        .downcast_ref::<MirPtrType>()
        .map(|mir_ptr| mir_ptr.pointee)
        .ok_or_else(|| {
            pliron::input_error!(
                loc.clone(),
                "mir.array_element_addr result must be a MIR pointer type; carrier element sizing has no fact to derive from"
            )
        })?;

    // The root carrier proof guarantees byte-faithfulness. Recheck the exact
    // projected stride here so a malformed private fact still fails closed at
    // its consumer instead of silently changing pointer arithmetic.
    let rustc_stride = mir_element_stride(ctx, semantic_element_ty);
    let llvm_size = llvm_type_size_align(ctx, carrier_element_ty).map(|(size, _)| size);
    if let (Some(stride), Some(llvm_size)) = (rustc_stride, llvm_size)
        && stride != llvm_size
    {
        return pliron::input_err_noloc!(
            "packed-AS3 carrier array element stride mismatch: rustc strides by {} bytes but the physical carrier element occupies {}",
            stride,
            llvm_size
        );
    }

    use llvm_export::ops::GepIndex;
    let element_align = carrier_projection_alignment(ctx, op).ok_or_else(|| {
        pliron::input_error_noloc!("carrier array projection is missing its verified alignment")
    })?;
    let gep = llvm::GetElementPtrOp::new_with_no_wrap_flags(
        ctx,
        arr_ptr,
        vec![GepIndex::Value(index)],
        carrier_element_ty,
        llvm_export::attributes::GepNoWrapFlags::INBOUNDS,
    );
    rewriter.insert_operation(ctx, gep.get_operation());
    llvm_export::ops::set_address_alignment(ctx, gep.get_operation(), element_align);
    rewriter.replace_operation(ctx, op, gep.get_operation());
    Ok(())
}
